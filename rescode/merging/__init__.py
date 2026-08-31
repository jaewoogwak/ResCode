"""ResCODE merging package: RESCODE clustering + ResCODE residual-code restoration.

Re-exports the public API of the residual-code coalescing modules so they
can be imported as `from rescode.merging import ...`.
"""

from .config import (
    RescodeConfig,
    PreActStarConfig,
    canonical_lowrank_target,
    parse_lowrank_target,
)
from .cluster_map import ExpertClusterMap, build_cluster_maps_from_group_state
from .meta_expert import (
    RescodeMetaExpert,
    PreActModule,
    compute_cluster_gate_and_rho,
)
from .convert_model import (
    convert_qwen_to_rescode,
    convert_qwen3_to_rescode,
    convert_mixtral_to_rescode,
    convert_deepseek_to_rescode,
)
from .save_load import save_rescode_state, load_rescode_state
from .diagnostics import (
    log_cluster_stats,
    compute_reconstruction_error,
    log_code_statistics,
    log_preact_statistics,
    log_giant_cluster_summary,
)
from .fit_codes import (
    LayerCalibration,
    OnlineTeacher,
    collect_teacher_signals_qwen,
    collect_teacher_signals_mixtral,
    collect_teacher_signals_deepseek,
    fit_codes_for_block,
    fit_all_codes,
    fit_codes_joint_lm,
    freeze_non_code_parameters,
)
from .preact_calib import collect_preact_stats
from .lowrank_init import (
    compute_covariance_init,
    compute_expert_svd_align_init,
    compute_lowrank_preact_init_data,
)
from .eval_summary import (
    summarize_task_metrics,
    format_summary,
    report_eval_summary,
)

__all__ = [
    "RescodeConfig",
    "PreActStarConfig",
    "canonical_lowrank_target",
    "parse_lowrank_target",
    "ExpertClusterMap",
    "build_cluster_maps_from_group_state",
    "RescodeMetaExpert",
    "PreActModule",
    "compute_cluster_gate_and_rho",
    "convert_qwen_to_rescode",
    "convert_qwen3_to_rescode",
    "convert_mixtral_to_rescode",
    "convert_deepseek_to_rescode",
    "save_rescode_state",
    "load_rescode_state",
    "log_cluster_stats",
    "compute_reconstruction_error",
    "log_code_statistics",
    "log_preact_statistics",
    "log_giant_cluster_summary",
    "LayerCalibration",
    "collect_teacher_signals_qwen",
    "collect_teacher_signals_mixtral",
    "collect_teacher_signals_deepseek",
    "fit_codes_for_block",
    "fit_all_codes",
    "freeze_non_code_parameters",
    "collect_preact_stats",
    "summarize_task_metrics",
    "format_summary",
    "report_eval_summary",
    "fit_codes_joint_lm",
    "compute_covariance_init",
    "compute_expert_svd_align_init",
    "compute_lowrank_preact_init_data",
    "OnlineTeacher",
]
