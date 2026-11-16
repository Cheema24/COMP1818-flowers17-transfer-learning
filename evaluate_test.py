import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from plot_results import plot_confusion, plot_test_confusions, plot_training_cost
from torch.utils.data import DataLoader
from flower_data import load_tensors
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

RUNS = [
    ("Frozen baseline", "baseline_20261112_030300_944563", "baseline_model.pth"),
    ("Full LP-FT", "two_stage_20261112_030318_406056", "two_stage_model.pth"),
    ("Final-block extension", "final_block_20261112_030815_639295", "final_block_model.pth"),
]


def save_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_comparison(rows, output, runs):
    colors = ["#526d82", "#db8551", "#16877b"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    names = ["Frozen\nbaseline", "Full\nLP-FT", "Final-block\nextension"]
    for ax, key, title in zip(axes, ["test_accuracy", "test_macro_f1"], ["Test accuracy", "Test macro-F1"]):
        values = [100 * row[key] for row in rows]
        bars = ax.bar(names, values, color=colors, width=0.65)
        ax.set(title=title, ylim=(0, 105), ylabel="Percent")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
    fig.suptitle("Flowers-17: fixed-checkpoint test comparison | 340 images | split 1", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "test_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (name, folder, _), row in zip(axes, runs, rows):
        with (ROOT / "results" / folder / "history.csv").open() as file:
            history = list(csv.DictReader(file))
        epochs = [int(item["epoch"]) for item in history]
        accuracy = [100 * float(item["val_accuracy"]) for item in history]
        ax.plot(epochs, accuracy, marker="o", markersize=3, color=colors[rows.index(row)])
        ax.set(title=name, xlabel="Stage-1 epoch" if name == "Frozen baseline" else "Stage-2 epoch",
               ylim=(50, 100), ylabel="Validation accuracy (%)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.03, 0.05, "Validation history; not test epochs", transform=ax.transAxes, fontsize=8)
    fig.suptitle("Learning history used before test evaluation (50-100% axis)")
    fig.tight_layout()
    fig.savefig(output / "validation_learning_curves.png", dpi=180)
    plt.close(fig)


# 001364457Cheema: Evaluate saved models without training.

def main():
    parser = argparse.ArgumentParser(description="Evaluate three saved flower classifiers.")
    parser.add_argument("--baseline", type=Path, help="Completed baseline run directory.")
    parser.add_argument("--full", type=Path, help="Completed full fine-tuning run directory.")
    parser.add_argument("--extension", type=Path, help="Completed final-block run directory.")
    args = parser.parse_args()
    supplied = [args.baseline, args.full, args.extension]
    if any(p is not None for p in supplied) and not all(p is not None for p in supplied):
        parser.error("Supply --baseline, --full and --extension together, or omit all to evaluate the original runs.")
    runs = [(name, str(path.resolve()), filename)
            for (name, _, filename), path in zip(RUNS, supplied)] if all(p is not None for p in supplied) else RUNS
    for _, folder, filename in runs:
        for required in [filename, "summary.json", "config.json", "split_ids.json", "history.csv"]:
            if not (ROOT / "results" / folder / required).is_file():
                parser.error(f"Missing completed-run file: {ROOT / 'results' / folder / required}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output = ROOT / "results" / datetime.now().strftime("test_evaluation_%Y%m%d_%H%M%S_%f")
    official = loadmat(ROOT / "data" / "datasplits.mat")
    ids = official["tst1"].ravel().astype(int).tolist()
    assert len(ids) == len(set(ids)) == 340
    assert not set(ids) & set(official["trn1"].ravel().tolist())
    assert not set(ids) & set(official["val1"].ravel().tolist())
    for _, folder, _ in runs:
        saved = json.loads((ROOT / "results" / folder / "split_ids.json").read_text())
        for name, key in [("train", "trn1"), ("validation", "val1"), ("test", "tst1")]:
            expected = official[key].ravel().astype(int).tolist()
            if len(saved[name]) != len(expected) or set(saved[name]) != set(expected):
                parser.error(f"{folder}: {name} IDs do not match official split 1.")
        if saved["test"] != ids:
            parser.error(f"{folder}: test ID order does not match official split 1.")
    output.mkdir(parents=True)
    (output / "test_image_ids.json").write_text(json.dumps(ids, indent=2))
    dataset = load_tensors(ids, MobileNet_V3_Small_Weights.DEFAULT.transforms())
    assert torch.bincount(dataset.tensors[1], minlength=17).tolist() == [20] * 17
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    labels = dataset.tensors[1].tolist()
    metrics, manifests = [], []
    for name, folder, filename in runs:
        run = ROOT / "results" / folder
        checkpoint = run / filename
        source = json.loads((run / "summary.json").read_text())
        before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model = mobilenet_v3_small(weights=None)
        model.classifier = nn.Linear(model.classifier[0].in_features, 17)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        predictions, total_loss = [], 0.0
        started = time.perf_counter()
        with torch.inference_mode():
            for images, targets in loader:
                logits = model(images)
                total_loss += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
                predictions.extend(logits.argmax(1).tolist())
        seconds = time.perf_counter() - started
        matrix = confusion_matrix(labels, predictions, labels=list(range(17)))
        assert matrix.sum() == 340 and np.all(matrix.sum(axis=1) == 20)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_hash
        row = dict(method=name, test_accuracy=accuracy_score(labels, predictions),
                   test_macro_f1=f1_score(labels, predictions, labels=list(range(17)), average="macro", zero_division=0),
                   correct=int(np.trace(matrix)), incorrect=int(340-np.trace(matrix)),
                   test_loss=total_loss / 340,
                   validation_accuracy=source["validation_accuracy"],
                   trainable_parameters_during_training=source["trainable_parameters"],
                   test_forward_seconds=seconds)
        metrics.append(row)
        slug = "baseline" if name == "Frozen baseline" else "full_lpft" if name == "Full LP-FT" else "extension"
        save_csv(output / f"{slug}_test_predictions.csv",
                 [dict(image_id=i, true_class=y, predicted_class=p) for i, y, p in zip(ids, labels, predictions)])
        np.savetxt(output / f"{slug}_confusion_matrix.csv", matrix, delimiter=",", fmt="%d",
                   header="Rows=true class 0-16; columns=predicted class 0-16")
        report = classification_report(labels, predictions, labels=list(range(17)), output_dict=True, zero_division=0)
        (output / f"{slug}_classification_report.json").write_text(json.dumps(report, indent=2))
        plot_confusion(matrix, name + ": test confusion matrix", output / f"{slug}_confusion_matrix.png")
        manifests.append(dict(method=name, source_run=folder, checkpoint=filename,
                              sha256=before_hash, selected_epoch=source["best_epoch"]))
        print(json.dumps(row, indent=2), flush=True)
    save_csv(output / "test_comparison.csv", metrics)
    draw_comparison(metrics, output, runs)
    (output / "manifest.json").write_text(json.dumps(dict(split=1, test_images=340,
        images_per_class=20, labels="numeric classes 0-16; 80 sequential source images per class",
        preprocessing="ImageNet default resize 256, center crop 224, normalize",
        checkpoint_selection="completed on validation before test evaluation",
        checkpoints=manifests, training_performed=False, test_used_for_selection=False,
        trivial_majority_test_accuracy=1/17), indent=2))
    plot_test_confusions(output)
    plot_training_cost(output)
    lines = ["# Fixed-checkpoint test evaluation", "", "Official split 1: 340 test images, 20 per class. No training or checkpoint changes occurred.", "",
             "| Method | Test accuracy | Macro-F1 | Correct / 340 | Test loss |", "|---|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['method']} | {row['test_accuracy']:.2%} | {row['test_macro_f1']:.4f} | {row['correct']} | {row['test_loss']:.4f} |")
    lines += ["", "Rows in confusion matrices are true classes; columns are predictions. Labels are 0-16.",
              "Learning curves use validation data. Checkpoints were selected by validation loss.",
              "One split and one seed cannot establish superiority. Do not tune against test scores.",
              "Training times include stage 1. All deployed models have the same architecture."]
    (output / "TEST_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"Saved test evaluation to {output}", flush=True)


if __name__ == "__main__":
    main()
