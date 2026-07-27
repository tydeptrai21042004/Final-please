# One-cell Kaggle runner: Cross-Fitted Evidence Adaptive Tangent-Core V6 + compatible baselines.
# Kaggle settings: Internet ON; GPU T4 x2 recommended.

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ============================= EDITABLE CONFIGURATION =============================
REPO_URL = "https://github.com/tydeptrai21042004/TRSO_final.git"
REPO_REF = "main"
PREFER_UPLOADED_RELEASE = True
ALLOW_GITHUB_FALLBACK = True

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO = WORK / "TRSO"
DATA_PATH = WORK / "data"
OUTPUT_ROOT = WORK / "trso_cross_fitted_v6_one_seed_10epoch_results"

DATASET = "dtd"
TASK = "auto"
DOWNLOAD = True
DATASET_ARGS = {"dtd_partition": 1}

BACKBONES = "resnet50@torchvision,vit_tiny_patch16_224@timm"
METHODS = "linear,trso,full,lora,bitfit,ssf,adaptformer,conv"
SEEDS = "0"
EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 4
INPUT_SIZE = 0

PEFT_LR = 1e-3
FULL_LR = 1e-4
LINEAR_LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
MIN_LR = 1e-6
AUGMENTATION = "strong"
PEFT_HEAD_LR_SCALE = 0.5
PEFT_FREEZE_HEAD = False

RUN_TESTS = True
RUN_FAIRNESS_CHECK = True
KEEP_CHECKPOINTS = False
PROFILE_EFFICIENCY = True
MEASURE_EVAL_LATENCY = True
# ================================================================================


def run(command, *, cwd=None, env=None, check=True, capture=False):
    command = [str(value) for value in command]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        check=False,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture and result.stdout:
        print(result.stdout, end="")
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


def is_clean_release(root: Path) -> bool:
    required = [
        root / "main.py",
        root / "tools" / "run_fair_suite.py",
        root / "tools" / "verify_fairness.py",
        root / "tools" / "aggregate_revision_results.py",
        root / "models" / "tuning_modules" / "mdl_tangent_core.py",
        root / "tests" / "test_mdl_tangent_core.py",
        root / "tests" / "test_release_contract.py",
    ]
    forbidden = [
        root / "models" / "task_response_unified.py",
        root / "models" / "task_response_biaxial.py",
        root / "models" / "task_response_adapter.py",
        root / "models" / "tuning_modules" / "information_spectral.py",
        root / "models" / "tuning_modules" / "tangent_core.py",
        root / "tests" / "test_information_spectral.py",
        root / "tests" / "test_trso_v5_unified.py",
        root / "KAGGLE_TRSO_V5_V6_ONE_CELL.py",
        root / "KAGGLE_INFORMATION_SPECTRAL_ONE_CELL.py",
    ]
    return all(path.is_file() for path in required) and not any(path.exists() for path in forbidden)


def find_extracted_release():
    """Find a Kaggle dataset that was automatically extracted from an uploaded ZIP."""
    if not PREFER_UPLOADED_RELEASE or not INPUT.exists():
        return None
    candidates = []
    for main_file in INPUT.rglob("main.py"):
        root = main_file.parent
        if is_clean_release(root):
            candidates.append(root)
    if not candidates:
        return None
    candidates.sort(key=lambda path: (len(path.parts), len(str(path))))
    return candidates[0]


def find_uploaded_zips():
    """Return candidate ZIPs; Kaggle may retain archives or auto-extract them."""
    if not PREFER_UPLOADED_RELEASE or not INPUT.exists():
        return []
    candidates = []
    for path in INPUT.rglob("*.zip"):
        lower = path.name.lower()
        if "result" in lower or "summary" in lower:
            continue
        candidates.append(path)
    candidates.sort(
        key=lambda path: (
            0 if "information" in path.name.lower() or "mdl" in path.name.lower() or "tangent" in path.name.lower() or "final" in path.name.lower() else 1,
            0 if "trso" in path.name.lower() else 1,
            len(str(path)),
        )
    )
    return candidates


def locate_release_in_tree(extract_root: Path):
    for main_file in extract_root.rglob("main.py"):
        root = main_file.parent
        if is_clean_release(root):
            return root
    return None


def install_repository():
    if REPO.exists():
        shutil.rmtree(REPO)

    extracted = find_extracted_release()
    if extracted is not None:
        print("Using auto-extracted Kaggle release:", extracted)
        shutil.copytree(extracted, REPO, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        return {"kind": "kaggle_extracted", "source": str(extracted)}

    rejected_zips = []
    for uploaded_zip in find_uploaded_zips():
        print("Inspecting uploaded ZIP:", uploaded_zip)
        extract_root = WORK / "_trso_release_extract"
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True)
        try:
            with zipfile.ZipFile(uploaded_zip) as archive:
                archive.extractall(extract_root)
        except zipfile.BadZipFile:
            rejected_zips.append(f"{uploaded_zip} (invalid ZIP)")
            continue
        source_root = locate_release_in_tree(extract_root)
        if source_root is None:
            rejected_zips.append(f"{uploaded_zip} (legacy/unknown repository)")
            continue
        print("Using uploaded release ZIP:", uploaded_zip)
        shutil.copytree(source_root, REPO, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        return {"kind": "uploaded_zip", "source": str(uploaded_zip)}

    if rejected_zips:
        print("Rejected uploaded archives:")
        for item in rejected_zips:
            print(" -", item)

    if not ALLOW_GITHUB_FALLBACK:
        raise RuntimeError("No clean uploaded release was found under /kaggle/input.")

    print("No uploaded clean release found; cloning GitHub.")
    run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, REPO])
    commit = run(["git", "rev-parse", "HEAD"], cwd=REPO, capture=True).stdout.strip()
    if not is_clean_release(REPO):
        raise RuntimeError(
            "The cloned GitHub branch is not the clean Cross-Fitted Evidence Tangent-Core V6 release. "
            "Upload TRSO-mdl-tangent-v6-final-fixed.zip as a Kaggle dataset, "
            "or update GitHub before rerunning."
        )
    return {"kind": "git", "source": REPO_URL, "ref": REPO_REF, "commit": commit}


source_provenance = install_repository()

# Preserve Kaggle's CUDA-enabled PyTorch and install only missing dependencies.
run([
    sys.executable, "-m", "pip", "install", "-q",
    "timm>=0.9,<2",
    "pandas>=2",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "tqdm>=4.65",
    "pytest>=8",
])

help_text = run(
    [sys.executable, "-m", "tools.run_fair_suite", "--help"],
    cwd=REPO,
    capture=True,
).stdout
legacy_options = [
    "--trso_variant",
    "--trso_rank",
    "--trso_budget",
    "--trso_parameter_budget",
    "--trso_tangent_rank",
    "--trso_tangent_budget",
    "--trso_calibration_batches",
    "--trso_max_adapters",
    "--trso_auto_budget_ratio",
    "--trso_basis_trainable",
    "--trso_residual_target",
    "--trso_gain",
    "--trso_layers",
    "--trso_core_mode",
    "--trso_head_policy",
]
found_legacy = [option for option in legacy_options if option in help_text]
if found_legacy:
    raise RuntimeError(
        "A contaminated legacy runner was selected; removed proposal controls remain: "
        + ", ".join(found_legacy)
    )

import torch

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this cell.")

gpu_count = torch.cuda.device_count()
gpu_ids = ",".join(str(index) for index in range(gpu_count))
parallel_runs = max(1, min(gpu_count, 2))
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(gpu_count)])
print("Repository provenance:", source_provenance)

DATA_PATH.mkdir(parents=True, exist_ok=True)
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(parents=True)
manifest = OUTPUT_ROOT / "fair_manifest.json"

if RUN_TESTS:
    run([sys.executable, "-m", "pytest", "-q"], cwd=REPO)

fair_command = [
    sys.executable, "-m", "tools.run_fair_suite",
    "--dataset", DATASET,
    "--task", TASK,
    "--data_path", DATA_PATH,
    "--download", str(DOWNLOAD),
    "--dataset_args_json", json.dumps(DATASET_ARGS),
    "--backbones", BACKBONES,
    "--methods", METHODS,
    "--seeds", SEEDS,
    "--epochs", EPOCHS,
    "--batch_size", BATCH_SIZE,
    "--num_workers", NUM_WORKERS,
    "--input_size", INPUT_SIZE,
    "--peft_lr", PEFT_LR,
    "--full_lr", FULL_LR,
    "--linear_lr", LINEAR_LR,
    "--weight_decay", WEIGHT_DECAY,
    "--warmup_epochs", WARMUP_EPOCHS,
    "--min_lr", MIN_LR,
    "--augmentation", AUGMENTATION,
    "--peft_head_lr_scale", PEFT_HEAD_LR_SCALE,
    "--peft_freeze_head", str(PEFT_FREEZE_HEAD),
    "--output_root", OUTPUT_ROOT / "comparison",
    "--manifest", manifest,
    "--device", "cuda",
    "--gpu_ids", gpu_ids,
    "--parallel_runs", parallel_runs,
    "--profile_efficiency", str(PROFILE_EFFICIENCY),
    "--measure_eval_latency", str(MEASURE_EVAL_LATENCY),
    "--execute",
]
run(fair_command, cwd=REPO)

compatibility = manifest.with_name(manifest.stem + "_compatibility.json")
if RUN_FAIRNESS_CHECK:
    run([
        sys.executable, "-m", "tools.verify_fairness",
        "--manifest", manifest,
        "--compatibility", compatibility,
        "--output", OUTPUT_ROOT / "fairness_verification.json",
    ], cwd=REPO)

run([
    sys.executable, "-m", "tools.aggregate_revision_results",
    "--root", OUTPUT_ROOT,
    "--out_csv", OUTPUT_ROOT / "all_results.csv",
], cwd=REPO)

compact_script = OUTPUT_ROOT / "make_one_seed_table.py"
compact_script.write_text(
    r'''from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
frame = pd.read_csv(root / "all_results.csv")
preferred = [
    "experiment_suite", "experiment_name", "dataset", "task_type", "backbone",
    "test_acc1", "test_macro_f1", "test_loss", "test_ece",
    "param_trainable_params", "param_total_params", "param_trainable_ratio",
    "eff_latency_ms_per_image", "eff_fps", "eff_peak_inference_memory_mb",
]
columns = [column for column in preferred if column in frame.columns]
table = frame[columns].copy()
if "backbone" in table.columns and "test_acc1" in table.columns:
    table = table.sort_values(["backbone", "test_acc1"], ascending=[True, False])
table.to_csv(root / "one_seed_10epoch_comparison.csv", index=False)
print(table.to_string(index=False))
''',
    encoding="utf-8",
)
run([sys.executable, compact_script], cwd=REPO)

provenance = {
    **source_provenance,
    "dataset": DATASET,
    "dataset_args": DATASET_ARGS,
    "backbones": BACKBONES,
    "methods": METHODS,
    "seeds": SEEDS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
}
(OUTPUT_ROOT / "run_provenance.json").write_text(
    json.dumps(provenance, indent=2), encoding="utf-8"
)

if not KEEP_CHECKPOINTS:
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        for path in OUTPUT_ROOT.rglob(pattern):
            path.unlink(missing_ok=True)

zip_path = WORK / "trso_cross_fitted_v6_one_seed_10epoch_results.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT_ROOT.rglob("*"):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT_ROOT.parent))

print("\nCompleted Cross-Fitted Evidence Adaptive Tangent-Core V6 plus compatible baselines.")
print("Seed:", SEEDS)
print("Epochs:", EPOCHS)
print("Results directory:", OUTPUT_ROOT)
print("Raw results:", OUTPUT_ROOT / "all_results.csv")
print("Compact table:", OUTPUT_ROOT / "one_seed_10epoch_comparison.csv")
print("Fairness report:", OUTPUT_ROOT / "fairness_verification.json")
print("Downloadable ZIP:", zip_path)
