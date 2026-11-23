"""Paged experts over the MXFP4-packed routed experts of an FP8 checkpoint (DeepSeek-V4).

The methods are built without weights: the layer is only built, and paged experts decides
its format there, before the checkpoint is read. ``is_fp4_expert`` is still True at that
point on the dequantized path -- only the post-load step clears it -- so what decides the
format is ``dequant_fp4_to_fp8``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.moe.paged_experts import formats
from sglang.srt.layers.moe.paged_experts.formats import (
    Fp8BlockFormat,
    Fp8DequantFp4ExpertsFormat,
    format_for,
)
from sglang.srt.layers.moe.paged_experts.method import check_paged_experts_compat
from sglang.srt.layers.moe.paged_experts.runners import (
    TRITON,
    TRITON_REPACK,
    contract_for,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}
TOP_K = 6


def _method(*, fp4_experts: bool, dequant: bool):
    """An ``Fp8MoEMethod`` as the loader hands it to ``FusedMoE``: the checkpoint's fp8
    config with the DSV4 routed-expert layout filled in (see loader.py)."""
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

    config = Fp8Config.from_config(dict(QUANT_CONFIG))
    config.is_fp4_experts = fp4_experts
    config.is_dsv4_fp4_experts = fp4_experts
    config.dequant_fp4_to_fp8 = dequant
    return Fp8MoEMethod(config)


class TestFp4DequantFormat(CustomTestCase):
    def test_dequantized_experts_are_resolved_while_the_layer_is_built(self):
        """The check runs as the layer is built, when ``is_fp4_expert`` is still True: the
        format must be found from ``dequant_fp4_to_fp8`` alone, or the server refuses a
        checkpoint it can serve."""
        method = _method(fp4_experts=True, dequant=True)
        self.assertTrue(method.is_fp4_expert)  # cleared only after loading

        fmt = format_for(SimpleNamespace(scheme=None), method)

        self.assertIsInstance(fmt, Fp8DequantFp4ExpertsFormat)
        self.assertIn("dequantized", fmt.label)

    def test_packed_experts_without_the_env_are_refused_with_the_env_to_set(self):
        method = _method(fp4_experts=True, dequant=False)

        self.assertIsNone(format_for(SimpleNamespace(scheme=None), method))

        problems = self._compat(method)
        self.assertIn("SGLANG_DSV4_FP4_DEQUANT=1", problems)
        # The runner cannot be marlin with it, and marlin needs SM90/SM120.
        self.assertIn("auto", problems)
        self.assertIn("SM90", problems)

    def test_the_dequantized_format_repacks_the_experts_as_they_load(self):
        """Dequantization rewrites each expert, so the post-load step has to run over the
        host store K experts at a time: the Triton runner alone does not imply that."""
        method = _method(fp4_experts=True, dequant=True)
        runner = SimpleNamespace(runner_backend=_backend("TRITON"))

        contract = contract_for(runner, method)

        self.assertIs(contract, TRITON_REPACK)
        self.assertTrue(contract.repacks)

    def test_block_fp8_without_fp4_experts_is_unchanged(self):
        """A plain block-FP8 checkpoint keeps its format and its non-repacking contract."""
        method = _method(fp4_experts=False, dequant=False)
        layer = SimpleNamespace(scheme=None)

        self.assertIsInstance(format_for(layer, method), Fp8BlockFormat)
        self.assertNotIsInstance(format_for(layer, method), Fp8DequantFp4ExpertsFormat)
        self.assertIs(
            contract_for(SimpleNamespace(runner_backend=_backend("TRITON")), method),
            TRITON,
        )

    def test_every_chunk_is_dequantized_not_only_the_first(self):
        """The step clears ``is_fp4_expert`` on the method once it has dequantized its chunk.
        Each chunk's run starts from the method as it was before the first, or every chunk
        after the first pages weights that are still FP4-packed -- silently, since nothing
        checks the layout of the paged rows."""
        import torch

        from sglang.srt.layers.moe.paged_experts.formats import PostLoad

        class _Dequantizing:
            # Instance attributes, as Fp8MoEMethod sets them in __init__: the step restores
            # the method's instance state between chunks.
            def __init__(self):
                self.is_fp4_expert = True
                self.dequant_fp4_to_fp8 = True

            def process_weights_after_loading(self, layer):
                if self.is_fp4_expert:  # fp8.py: only while the experts are still FP4
                    layer.w13_weight.data = layer.w13_weight.data + 1000
                    self.is_fp4_expert = False

        class _Loads:
            num_slots, rows_like = 2, ("w13_weight",)

            def rows(self, experts, out=None):
                return {
                    "w13_weight": torch.stack(
                        [torch.full((2, 2), float(e)) for e in experts]
                    )
                }

        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w13_weight = torch.nn.Parameter(
                    torch.zeros(2, 2, 2), requires_grad=False
                )

        layer, method = _Layer(), _Dequantizing()
        step = PostLoad(layer, method, _Loads(), repacks=True, unpaged=())

        first = step.run([0, 1])["w13_weight"][0][0][0].item()
        second = step.run([2, 3])["w13_weight"][0][0][0].item()

        self.assertGreaterEqual(first, 1000)
        # 2.0 + 1000: dequantized too, not the raw 2.0 the second chunk was loaded with.
        self.assertGreaterEqual(second, 1000)

    def test_the_format_is_registered(self):
        self.assertIn(Fp8DequantFp4ExpertsFormat, formats.FORMATS)
        # Listed before the format it specializes, so the log names the dequantized case.
        self.assertLess(
            formats.FORMATS.index(Fp8DequantFp4ExpertsFormat),
            formats.FORMATS.index(Fp8BlockFormat),
        )

    def _compat(self, method) -> str:
        """The joined problems ``check_paged_experts_compat`` raises for ``method``."""
        layer = SimpleNamespace(top_k=TOP_K, num_fused_shared_experts=0, quant_method=method)
        method_path = "sglang.srt.layers.moe.paged_experts.method."
        moe = SimpleNamespace(
            enable_eplb=False,
            paged_experts_num_resident=TOP_K,
            enable_fused_moe_sum_all_reduce=False,
        )
        with (
            patch(
                method_path + "get_parallel",
                return_value=SimpleNamespace(tp_size=1, ep_size=1, pp_size=1, dp_size=1),
            ),
            patch(
                method_path + "get_exec",
                return_value=SimpleNamespace(
                    moe=moe,
                    features=SimpleNamespace(enable_memory_saver=False),
                    kernel=SimpleNamespace(enable_nvidia_megakernel=False),
                ),
            ),
            patch(
                method_path + "get_lora",
                return_value=SimpleNamespace(enable_lora=False),
            ),
            patch(
                method_path + "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch(
                method_path + "get_model",
                return_value=SimpleNamespace(load_format="auto"),
            ),
            patch(method_path + "check_cuda_graph_backend", return_value=True),
        ):
            with self.assertRaises(RuntimeError) as caught:
                check_paged_experts_compat(
                    layer=layer,
                    expert_format=format_for(SimpleNamespace(scheme=None), method),
                )
        return str(caught.exception)


def _backend(name: str):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    return MoeRunnerBackend[name]


if __name__ == "__main__":
    unittest.main()
