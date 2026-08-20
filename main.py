import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPT_MAP = {
    "train": {
        ("cbam", "ip102"): "eff_cbam.py",
        ("cbam", "do"): "eff_cbam_train_Do.py",
        ("cbam", "xie"): "eff_cbam_train_xie.py",
        ("baseline", "ip102"): "Ef2_original.py",
        ("eca", "ip102"): "eval_eff2_ECA.py",
        ("ml", "ip102"): "train_ml_model.py",
    },
    "eval": {
        ("cbam", "ip102"): "eval_cbam.py",
        ("cbam", "do"): "eval_cbam_do.py",
        ("cbam", "xie"): "eval_cbam_xie.py",
        ("baseline", "ip102"): "eval_eff2_original.py",
        ("eca", "ip102"): "Eff2_ECA.py",
    },
}

ALIASES = {
    "ip": "ip102",
    "do_dataset": "do",
}


def normalize_dataset(dataset: str) -> str:
    dataset = dataset.strip().lower()
    return ALIASES.get(dataset, dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entry point for train/eval scripts in project."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train", "eval"],
        help="Choose task: train or eval.",
    )
    parser.add_argument(
        "--model",
        default="cbam",
        choices=["cbam", "eca", "baseline", "ml"],
        help="Choose model/pipeline.",
    )
    parser.add_argument(
        "--dataset",
        default="ip102",
        help="Dataset: ip102, do, xie (support alias: ip).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all supported mode/model/dataset combinations and exit.",
    )
    return parser


def print_supported_combinations() -> None:
    print("Supported combinations:")
    for mode, mapping in SCRIPT_MAP.items():
        for (model, dataset), script_name in sorted(mapping.items()):
            print(f"  {mode:5s} | model={model:8s} | dataset={dataset:5s} -> {script_name}")


def resolve_script(mode: str, model: str, dataset: str) -> Path:
    key = (model, dataset)
    script_name = SCRIPT_MAP.get(mode, {}).get(key)
    if script_name is None:
        print(
            f"Unsupported combination: mode={mode}, model={model}, dataset={dataset}.",
            file=sys.stderr,
        )
        print_supported_combinations()
        raise SystemExit(2)
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        raise SystemExit(2)
    return script_path


def run_script(script_path: Path) -> int:
    cmd = [sys.executable, str(script_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_supported_combinations()
        return 0

    if args.mode is None:
        parser.error("mode is required unless --list is provided")

    dataset = normalize_dataset(args.dataset)
    script_path = resolve_script(args.mode, args.model, dataset)
    return run_script(script_path)


if __name__ == "__main__":
    raise SystemExit(main())
