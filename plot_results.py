import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion(matrix, title, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=20)
    ax.set(xticks=range(17), yticks=range(17), xlabel="Predicted class",
           ylabel="True class", title=title)
    ax.tick_params(labelsize=9)
    for true in range(17):
        for predicted in range(17):
            count = int(matrix[true, predicted])
            if count:
                ax.text(predicted, true, str(count), ha="center", va="center",
                        color="white" if count > 10 else "#1d3557", fontsize=9)
    fig.colorbar(image, ax=ax, label="Images (20 per true class)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_training(folder):
    folder = Path(folder)
    with (folder / "history.csv").open() as file:
        history = list(csv.DictReader(file))
    summary = json.loads((folder / "summary.json").read_text())
    config = json.loads((folder / "config.json").read_text())
    title = config.get("method", "Frozen-feature linear baseline")
    stage = "Stage-1 epoch" if folder.name.startswith("baseline_") else "Stage-2 epoch"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, keys, ylabel in [(axes[0], ["train_loss", "val_loss"], "Cross-entropy loss"),
                              (axes[1], ["train_accuracy", "val_accuracy"], "Accuracy (%)")]:
        for key in keys:
            points = [(int(row["epoch"]), float(row[key])) for row in history if row.get(key)]
            if not points:
                continue
            x, y = zip(*points)
            multiplier = 100 if "accuracy" in key else 1
            ax.plot(x, [v * multiplier for v in y], marker="o", markersize=3,
                    label="Training" if key.startswith("train") else "Validation")
        ax.axvline(summary["best_epoch"], color="#777777", linestyle="--", label="Saved epoch")
        ax.set(xlabel=stage, ylabel=ylabel)
        if "Accuracy" in ylabel:
            ax.set_ylim(0, 105)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.text(0.5, 0.01, "Saved epoch selected by validation loss. Fine-tuning training loss is measured during updates.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(folder / "training_graphs.png", dpi=180)
    plt.close(fig)
    with (folder / "validation_predictions.csv").open() as file:
        predictions = list(csv.DictReader(file))
    matrix = confusion_matrix([int(r["true_class"]) for r in predictions],
                              [int(r["predicted_class"]) for r in predictions], labels=list(range(17)))
    np.savetxt(folder / "validation_confusion_matrix.csv", matrix, fmt="%d", delimiter=",",
               header="Rows=true class 0-16; columns=predicted class 0-16")
    plot_confusion(matrix, title + "\nValidation confusion matrix", folder / "validation_confusion_matrix.png")


def plot_test_confusions(folder):
    folder = Path(folder)
    names = ["Frozen baseline", "Full LP-FT", "Final-block extension"]
    slugs = ["baseline", "full_lpft", "extension"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), layout="constrained")
    for ax, name, slug in zip(axes, names, slugs):
        matrix = np.loadtxt(folder / f"{slug}_confusion_matrix.csv", delimiter=",", dtype=int)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=20)
        ax.set(xticks=range(17), yticks=range(17), xlabel="Predicted class", ylabel="True class",
               title=f"{name}\n{matrix.trace()/matrix.sum():.2%} | {matrix.trace()}/340 correct")
        ax.tick_params(labelsize=8)
        for true in range(17):
            for predicted in range(17):
                count = int(matrix[true, predicted])
                if count:
                    ax.text(predicted, true, str(count), ha="center", va="center", fontsize=8,
                            color="white" if count > 10 else "#1d3557")
    fig.colorbar(image, ax=axes.tolist(), shrink=0.75, label="Images (20 per true class)")
    fig.suptitle("Test confusion matrices | rows = true class; columns = predicted class | numeric labels 0-16")
    fig.savefig(folder / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def plot_training_cost(folder):
    with (Path(folder) / "manifest.json").open() as file:
        runs = json.load(file)["checkpoints"]
    summaries = [json.loads((ROOT / "results" / run["source_run"] / "summary.json").read_text()) for run in runs]
    names = ["Frozen\nbaseline", "Full\nLP-FT", "Final-block\nextension"]
    colors = ["#526d82", "#db8551", "#16877b"]
    values = [[s["trainable_parameters"] for s in summaries],
              [s.get("total_two_stage_seconds", s.get("total_seconds")) for s in summaries]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, numbers, title, label in zip(axes, values,
            ["Trainable parameters", "Recorded CPU training + preparation time"], ["Parameters", "Seconds"]):
        bars = ax.bar(names, numbers, color=colors)
        ax.set(title=title, ylabel=label, ylim=(0, max(numbers) * 1.2))
        ax.spines[["top", "right"]].set_visible(False)
        for bar, number in zip(bars, numbers):
            ax.text(bar.get_x()+bar.get_width()/2, number+max(numbers)*0.02,
                    f"{number:,.0f}" if label == "Parameters" else f"{number:.1f}", ha="center")
    fig.text(0.5, 0.01, "Separate runs; stage 1 included for both fine-tuning methods. All deployed models have the same architecture.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(Path(folder) / "training_cost_comparison.png", dpi=180)
    plt.close(fig)


def main():
    # 001364457Cheema: Rebuild graphs from saved results.
    for folder in sorted((ROOT / "results").iterdir()):
        if (folder / "ABORTED.txt").exists():
            continue
        if (folder / "history.csv").exists() and (folder / "summary.json").exists():
            plot_training(folder)
            print(f"Saved training figures: {folder.name}", flush=True)
        if (folder / "test_comparison.csv").exists():
            plot_test_confusions(folder)
            plot_training_cost(folder)
            print(f"Saved comparison figures: {folder.name}", flush=True)


if __name__ == "__main__":
    main()
