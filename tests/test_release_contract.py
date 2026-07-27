from pathlib import Path
from types import SimpleNamespace

from main import get_args_parser
from tools.run_fair_suite import method_variant


ROOT = Path(__file__).resolve().parents[1]


def test_active_cli_exposes_no_removed_proposal_controls():
    actions = {action.dest for action in get_args_parser()._actions}
    removed = {
        "trso_variant", "trso_rank", "trso_budget", "trso_parameter_budget",
        "trso_tangent_rank", "trso_tangent_budget", "trso_calibration_batches",
        "trso_residual_target", "trso_gain", "trso_layers",
    }
    assert actions.isdisjoint(removed)
    assert "trso_fast_inference" in actions


def test_fair_runner_adds_only_deployment_control_for_trso():
    args = SimpleNamespace(ra_pretrained_checkpoint="")
    row = method_variant("trso", "cnn", 0, None, args)
    assert row == {
        "tuning_method": "trso",
        "seed": 0,
        "keep_pretrained_head": False,
        "trso_fast_inference": True,
    }


def test_removed_legacy_implementation_files_are_absent():
    removed = [
        "models/task_response_unified.py",
        "models/task_response_biaxial.py",
        "models/task_response_adapter.py",
        "models/tuning_modules/tangent_core.py",
        "KAGGLE_TRSO_V5_V6_ONE_CELL.py",
    ]
    assert all(not (ROOT / relative).exists() for relative in removed)


def test_only_mdl_evidence_proposal_module_is_active():
    text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "models.tuning_modules.mdl_tangent_core" in text
    assert "models.tuning_modules.information_spectral" not in text
    assert "task_response_unified" not in text
    assert not (ROOT / "models/tuning_modules/information_spectral.py").exists()
