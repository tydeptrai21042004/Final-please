"""Paper-reproduction tuning modules."""
from .prompter import (
    FixedPatchPrompter,
    PadPrompter,
    RandomPatchPrompter,
    VisualPromptingClassifier,
    build_prompter,
)
from .conv_adapter import ConvAdapter, ConvAdapterBottleneck, apply_conv_adapter_resnet50
from .program_module import ProgramModule
from .ssf import SSF, SSFPost, SSFMultiheadAttention, apply_ssf, merge_ssf_
from .bam_adapter import BAM, BAMAdapter, BAMResNet50
from .lora_transformer import (
    LoRALinear,
    LoRAQKVLinear,
    LoRAMultiheadAttention,
    apply_lora_transformer,
)
from .residual_adapter import ResidualAdapterResNet26
from .side_tuning import ConvSideNetwork, SideTuningClassifier
from .bitfit import set_bitfit_trainability
from .adaptformer import AdaptFormerAdapter, apply_adaptformer, set_adaptformer_trainability
from .piggyback import (
    BinaryMaskSTE, PiggybackConv2d, PiggybackLinear, apply_piggyback,
    set_piggyback_trainability, piggyback_storage, export_binary_masks,
)


def set_tuning_config(tuning_method, args):
    method = str(tuning_method).strip().lower().replace("-", "_")
    aliases = {
        "conv_adapter": "conv",
        "adapter": "conv",
        "task_response": "trso",
        "trso_adapter": "trso",
        "bam_adapter": "bam",
        "residual_adapter": "residual",
        "side_tuning": "sidetune",
        "adapt_former": "adaptformer",
        "piggy_back": "piggyback",
    }
    method = aliases.get(method, method)
    if method == "prompt":
        return {
            "method": method,
            "prompt_size": getattr(args, "prompt_size", 30),
            "prompt_type": getattr(args, "prompt_type", "padding"),
        }
    if method == "conv":
        return {
            "method": method,
            "kernel_size": getattr(args, "kernel_size", 3),
            "mode": getattr(args, "conv_adapter_mode", "conv_parallel"),
            "width": getattr(args, "adapt_size", 8),
            "scale": getattr(args, "adapt_scale", 1.0),
        }
    if method == "trso":
        return {"method": method, "proposal": "mdl_evidence_adaptive_tangent_core_v6"}
    if method == "bam":
        return {"method": method, "reduction": getattr(args, "bam_reduction", 16), "dilation": getattr(args, "bam_dilation", 4)}
    if method == "residual":
        return {"method": method, "mode": getattr(args, "ra_mode", "parallel")}
    if method == "ssf":
        return {"method": method, "init_std": getattr(args, "ssf_init_std", 0.02)}
    if method == "lora":
        return {"method": method, "rank": getattr(args, "lora_r", 8), "alpha": getattr(args, "lora_alpha", 16.0)}
    if method == "sidetune":
        return {
            "method": method,
            "alpha": getattr(args, "sidetune_alpha", 0.5),
            "side_arch": getattr(args, "sidetune_arch", "lightweight"),
            "side_width": getattr(args, "sidetune_width", 64),
            "side_depth": getattr(args, "sidetune_depth", 4),
        }
    if method == "bitfit":
        return {
            "method": method,
            "bias_scope": getattr(args, "bitfit_bias_scope", "all"),
            "train_head": getattr(args, "bitfit_train_head", True),
        }
    if method == "adaptformer":
        return {
            "method": method,
            "bottleneck": getattr(args, "adaptformer_dim", 16),
            "scale": getattr(args, "adaptformer_scale", 0.1),
            "dropout": getattr(args, "adaptformer_dropout", 0.0),
        }
    if method == "piggyback":
        return {
            "method": method,
            "threshold": getattr(args, "piggyback_threshold", 5e-3),
            "mask_init": getattr(args, "piggyback_mask_init", "ones"),
            "mask_linear": getattr(args, "piggyback_mask_linear", False),
        }
    if method in {"full", "linear"}:
        return {"method": method}
    if method in {"lora_conv", "lora_conv2d"}:
        raise NotImplementedError(
            "LoRA-Conv is an experimental repository control, not an original-paper baseline, "
            "and is excluded from strict reproduction runs."
        )
    raise NotImplementedError(f"Unknown tuning_method: {tuning_method}")


from .mdl_tangent_core import (
    MDLTangentCoreParametrization,
    MDLTangentRecord,
    MDLTangentReport,
    calibrate_mdl_tangent_core,
    iter_mdl_tangent_parametrizations,
    mdl_tangent_basis_value_count,
    mdl_tangent_parameter_count,
    merge_mdl_tangent_cores_,
    set_mdl_tangent_trainability,
)

__all__ = [name for name in globals() if not name.startswith("_")]
