import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from plot_results import plot_training
from torch.utils.data import DataLoader, TensorDataset
from flower_data import FlowerImages
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parent


def extract_features(model, loader, name):
    features, labels = [], []
    model.eval()  # 001364457Cheema: Keep frozen BatchNorm statistics unchanged.
    with torch.no_grad():
        for batch, (images, targets) in enumerate(loader, start=1):
            features.append(model(images))
            labels.append(targets)
            if batch % 5 == 0 or batch == len(loader):
                print(f"{name}: extracted batch {batch}/{len(loader)}", flush=True)
    return torch.cat(features), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(description="Train the frozen-feature baseline.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size and learning rate must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    output = ROOT / "results" / datetime.now().strftime("baseline_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)

    # 001364457Cheema: Use official split 1 and check for overlap.
    splits = loadmat(ROOT / "data" / "datasplits.mat")
    ids = {name: splits[key].ravel().astype(int) for name, key in
           [("train", "trn1"), ("validation", "val1"), ("test", "tst1")]}
    groups = [set(values.tolist()) for values in ids.values()]
    assert sum(map(len, groups)) == len(set.union(*groups)) == 1360
    assert set.union(*groups) == set(range(1, 1361))
    for name, values in ids.items():
        counts = np.bincount((values - 1) // 80, minlength=17)
        expected = 40 if name == "train" else 20
        assert np.all(counts == expected), (name, counts)
    print("CPU | official split 1 | train 680 | validation 340 | test untouched", flush=True)
    config = dict(vars(args), split=1, device="cpu", classes=17,
                  preprocessing="ImageNet default resize 256 / center crop 224",
                  augmentation=False, selection="lowest validation cross-entropy",
                  image_loading="requested compressed image bytes cached from local archive",
                  label_rule="(image_id - 1) // 80")
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "split_ids.json").write_text(json.dumps({k: v.tolist() for k, v in ids.items()}, indent=2))

    # 001364457Cheema: Load pretrained weights and remove the old classifier.
    model = mobilenet_v3_small(weights=None)
    state = torch.load(ROOT / "models" / "mobilenet_v3_small-imagenet.pth",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad = False
    feature_count = model.classifier[0].in_features
    model.classifier = nn.Identity()
    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()

    # 001364457Cheema: Extract train and validation features once.
    features = {}
    extraction_started = time.perf_counter()
    for name in ["train", "validation"]:
        loader = DataLoader(FlowerImages(ids[name], transform),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
        features[name] = extract_features(model, loader, name)
    extraction_seconds = time.perf_counter() - extraction_started
    train_x, train_y = features["train"]
    val_x, val_y = features["validation"]

    # 001364457Cheema: Train the new linear classifier.
    head = nn.Linear(feature_count, 17)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size,
                        shuffle=True, generator=generator)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = loss_function(head(inputs), labels)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            train_logits, val_logits = head(train_x), head(val_x)
            train_loss = loss_function(train_logits, train_y).item()
            val_loss = loss_function(val_logits, val_y).item()
            train_accuracy = (train_logits.argmax(1) == train_y).float().mean().item()
            val_accuracy = (val_logits.argmax(1) == val_y).float().mean().item()
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_accuracy=train_accuracy, val_accuracy=val_accuracy))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        print(f"Epoch {epoch:02d}/{args.epochs}: train loss {train_loss:.4f}, "
              f"val loss {val_loss:.4f}, val accuracy {val_accuracy:.2%}", flush=True)
    training_seconds = time.perf_counter() - training_started

    # 001364457Cheema: Save the best classifier and frozen features.
    head.load_state_dict(best_state)
    model.classifier = head
    model.eval()
    with torch.no_grad():
        predictions = head(val_x).argmax(1).numpy()
    majority_class = int(torch.bincount(train_y, minlength=17).argmax())
    summary = dict(best_epoch=best_epoch, validation_loss=best_loss,
                   validation_accuracy=accuracy_score(val_y.numpy(), predictions),
                   validation_macro_f1=f1_score(val_y.numpy(), predictions, average="macro", zero_division=0),
                   majority_validation_accuracy=float((val_y.numpy() == majority_class).mean()),
                   trainable_parameters=sum(p.numel() for p in head.parameters()),
                   feature_extraction_seconds=extraction_seconds,
                   head_training_seconds=training_seconds,
                   total_seconds=time.perf_counter() - started,
                   test_evaluated=False, seed=args.seed)
    torch.save(model.state_dict(), output / "baseline_model.pth")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output / "validation_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "true_class", "predicted_class"])
        writer.writerows(zip(ids["validation"].tolist(), val_y.tolist(), predictions.tolist()))
    plot_training(output)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved model and results to {output}", flush=True)


if __name__ == "__main__":
    main()
