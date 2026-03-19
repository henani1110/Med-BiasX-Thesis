import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_history_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "eval_score": float(row["eval_score"]),
                }
            )
    return rows


def infer_paths(input_path):
    if os.path.isdir(input_path):
        run_dir = input_path
        csv_path = os.path.join(run_dir, "history.csv")
        out_path = os.path.join(run_dir, "training_curves.png")
    else:
        csv_path = input_path
        run_dir = os.path.dirname(csv_path)
        out_path = os.path.join(run_dir, "training_curves.png")
    return csv_path, out_path


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from history.csv")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to run directory or history.csv file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output png path (default: <run_dir>/training_curves.png)",
    )
    args = parser.parse_args()

    csv_path, default_out = infer_paths(args.input)
    out_path = args.output if args.output else default_out

    if not os.path.exists(csv_path):
        raise FileNotFoundError("history.csv not found: {}".format(csv_path))

    rows = load_history_csv(csv_path)
    if not rows:
        raise RuntimeError("history.csv is empty: {}".format(csv_path))

    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    eval_score = [r["eval_score"] for r in rows]

    best_idx = max(range(len(eval_score)), key=lambda i: eval_score[i])
    best_epoch = epochs[best_idx]
    best_score = eval_score[best_idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(epochs, train_loss, marker="o", linewidth=1.5)
    axes[0].set_ylabel("Train Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, marker="o", linewidth=1.5)
    axes[1].set_ylabel("Train Acc")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, eval_score, marker="o", linewidth=1.5, label="Eval Score")
    axes[2].scatter([best_epoch], [best_score], color="red", zorder=3, label="Best")
    axes[2].set_ylabel("Eval Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Training Curves")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)

    print("Saved figure to {}".format(out_path))
    print("Best eval score: {:.4f} at epoch {}".format(best_score, best_epoch))


if __name__ == "__main__":
    main()
