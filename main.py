import argparse
import json
import os

from framework import load_framework_config


ALIASES = {
    "ip": "ip102",
    "do_dataset": "do",
}


def normalize_dataset(dataset: str) -> str:
    dataset = dataset.strip().lower()
    return ALIASES.get(dataset, dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified framework for train/eval/feature+SVM.")
    parser.add_argument(
        "mode",
        choices=["train", "eval", "svm", "full"],
        help="train: train model, eval: evaluate, svm: backbone feature + SVM, full: train+eval+svm",
    )
    parser.add_argument(
        "--dataset",
        default="ip102",
        choices=["ip102", "do", "xie", "ip", "do_dataset"],
        help="Dataset name. If --config is not set, auto-load corresponding config YAML.",
    )
    parser.add_argument(
        "--model",
        default="baseline",
        choices=["baseline", "cbam"],
        help="Backbone model type.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Custom config YAML path (optional).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. If omitted, use framework default output path.",
    )
    return parser


def print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=True, indent=2))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset = normalize_dataset(args.dataset)
    cfg = load_framework_config(dataset_name=dataset, config_path=args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if args.mode == "train":
        from framework.pipeline import train_pipeline

        result = train_pipeline(cfg=cfg, model_type=args.model, checkpoint_name=None)
        print_json(
            {
                "status": "ok",
                "mode": "train",
                "dataset": dataset,
                "model": args.model,
                "checkpoint_path": result.checkpoint_path,
                "best_val_accuracy": result.best_val_accuracy,
            }
        )
        return 0

    if args.mode == "eval":
        from framework.pipeline import evaluate_pipeline

        metrics = evaluate_pipeline(cfg=cfg, model_type=args.model, checkpoint_path=args.checkpoint)
        print_json(
            {
                "status": "ok",
                "mode": "eval",
                "dataset": dataset,
                "model": args.model,
                "metrics": metrics,
            }
        )
        return 0

    if args.mode == "svm":
        from framework.pipeline import run_feature_svm_pipeline

        metrics = run_feature_svm_pipeline(cfg=cfg, model_type=args.model, checkpoint_path=args.checkpoint)
        print_json(
            {
                "status": "ok",
                "mode": "svm",
                "dataset": dataset,
                "model": args.model,
                "metrics": metrics,
            }
        )
        return 0

    from framework.pipeline import run_full_pipeline

    summary = run_full_pipeline(cfg=cfg, model_type=args.model)
    print_json(
        {
            "status": "ok",
            "mode": "full",
            "dataset": dataset,
            "model": args.model,
            "summary": summary,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
