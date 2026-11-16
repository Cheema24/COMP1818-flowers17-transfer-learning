import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from plot_results import plot_training
from torch.utils.data import DataLoader
from flower_data import ROOT, load_tensors
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@torch.no_grad()
def validate(model, loader, loss_function):
    model.eval()
    total_loss, targets, predictions = 0.0, [], []
    for images, labels in loader:
        scores = model(images)
        total_loss += loss_function(scores, labels).item() * len(labels)
        targets.extend(labels.tolist())
        predictions.extend(scores.argmax(1).tolist())
    return dict(loss=total_loss / len(targets),
                accuracy=accuracy_score(targets, predictions),
                macro_f1=f1_score(targets, predictions, average="macro", zero_division=0)), targets, predictions


def main(final_block_only=False):
    description = "Fine-tune the final feature block." if final_block_only else "Fine-tune the full model."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "results" / "baseline_20261112_030300_944563")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many epochs without lower validation loss; 0 disables stopping.")
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("Epochs and learning rate must be positive.")
    if args.patience < 0:
        parser.error("Patience must be nonnegative.")
    baseline = args.baseline.resolve()
    baseline_config = json.loads((baseline / "config.json").read_text())
    baseline_summary = json.loads((baseline / "summary.json").read_text())
    ids = json.loads((baseline / "split_ids.json").read_text())
    groups = [set(ids[name]) for name in ["train", "validation", "test"]]
    assert sum(map(len, groups)) == len(set.union(*groups)) == 1360
    assert set.union(*groups) == set(range(1, 1361))
    seed = baseline_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    prefix = "final_block" if final_block_only else "two_stage"
    output = ROOT / "results" / datetime.now().strftime(prefix + "_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)

    # 001364457Cheema: Load the saved baseline.
    checkpoint = baseline / "baseline_model.pth"
    model = mobilenet_v3_small(weights=None)
    model.classifier = nn.Linear(model.classifier[0].in_features, 17)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    # 001364457Cheema: Choose which layers to train.
    for parameter in model.parameters():
        parameter.requires_grad = not final_block_only
    if final_block_only:
        for layer in [model.features[-1], model.classifier]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    checkpoint_name = "final_block_model.pth" if final_block_only else "two_stage_model.pth"
    method = "linear probing then final-block fine-tuning" if final_block_only else "linear probing then full fine-tuning"
    loss_function = nn.CrossEntropyLoss()
    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    datasets = {name: load_tensors(ids[name], transform) for name in ["train", "validation"]}
    train_loader = DataLoader(datasets["train"],
                              batch_size=baseline_config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(datasets["validation"],
                            batch_size=baseline_config["batch_size"], shuffle=False, num_workers=0)
    initial, _, _ = validate(model, val_loader, loss_function)
    assert abs(initial["accuracy"] - baseline_summary["validation_accuracy"]) < 1e-10
    assert abs(initial["loss"] - baseline_summary["validation_loss"]) < 1e-5
    print(f"Stage 1 checkpoint verified: validation accuracy {initial['accuracy']:.2%}", flush=True)
    print(f"Stage 2: {method} | CPU | same data | no augmentation | test untouched", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)

    config = dict(method=method, seed=seed,
                  final_feature_block_index=len(model.features) - 1,
                  trainable_parameter_names=[name for name, p in model.named_parameters() if p.requires_grad],
                  epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience,
                  batch_size=baseline_config["batch_size"], device="cpu",
                  split=baseline_config["split"], augmentation=False,
                  baseline_run=str(baseline),
                  baseline_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  stage1_best_epoch=baseline_summary["best_epoch"],
                  batchnorm=("only final feature block updates statistics; earlier blocks stay in eval mode"
                             if final_block_only else "running statistics updated during stage-2 training"),
                  image_loading="deterministic preprocessing once from local archive into memory",
                  selection="lowest validation loss among fine-tuned epochs 1..N",
                  paper="https://arxiv.org/abs/2202.10054")
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "split_ids.json").write_text(json.dumps(ids, indent=2))
    history = [dict(epoch=0, train_loss=None, val_loss=initial["loss"],
                    val_accuracy=initial["accuracy"], val_macro_f1=initial["macro_f1"])]
    best_loss, best_epoch, best_metrics = float("inf"), None, None
    best_targets, best_predictions = None, None
    epochs_without_improvement = 0
    stage2_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if final_block_only:
            # 001364457Cheema: Keep earlier BatchNorm layers frozen.
            model.features.eval()
            model.features[-1].train()
            model.classifier.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            scores = model(images)
            loss = loss_function(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        metrics, targets, predictions = validate(model, val_loader, loss_function)
        train_loss = total_loss / len(train_loader.dataset)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=metrics["loss"],
                            val_accuracy=metrics["accuracy"], val_macro_f1=metrics["macro_f1"]))
        if metrics["loss"] < best_loss:
            epochs_without_improvement = 0
            best_loss, best_epoch, best_metrics = metrics["loss"], epoch, metrics
            best_targets, best_predictions = targets, predictions
            torch.save(model.state_dict(), output / checkpoint_name)
        else:
            epochs_without_improvement += 1
        print(f"Fine-tune {epoch:02d}/{args.epochs}: train loss {train_loss:.4f}, "
              f"val loss {metrics['loss']:.4f}, val accuracy {metrics['accuracy']:.2%}", flush=True)
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Stopped after {args.patience} epochs without validation-loss improvement.", flush=True)
            break
    stage2_seconds = time.perf_counter() - stage2_started
    summary = dict(best_epoch=best_epoch, epochs_completed=epoch,
                   stopped_early=epoch < args.epochs, validation_loss=best_metrics["loss"],
                   validation_accuracy=best_metrics["accuracy"], validation_macro_f1=best_metrics["macro_f1"],
                   baseline_validation_accuracy=initial["accuracy"],
                   accuracy_change_percentage_points=100 * (best_metrics["accuracy"] - initial["accuracy"]),
                   trainable_parameters=sum(p.numel() for p in trainable),
                   stage2_seconds=stage2_seconds,
                   stage1_recorded_seconds=baseline_summary["total_seconds"],
                   total_two_stage_seconds=baseline_summary["total_seconds"] + time.perf_counter() - started,
                   test_evaluated=False, seed=seed)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output / "validation_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "true_class", "predicted_class"])
        writer.writerows(zip(ids["validation"], best_targets, best_predictions))
    plot_training(output)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
