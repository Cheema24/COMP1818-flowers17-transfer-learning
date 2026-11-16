import hashlib
import io
import json
import platform

import torch
import torchvision
from PIL import Image
from torchvision.models import mobilenet_v3_small

from flower_data import ROOT, FlowerImages


def main():
    print(f"Python {platform.python_version()}, torch {torch.__version__}, torchvision {torchvision.__version__}")
    torch.set_num_threads(4)

    # 001364457Cheema: Check the downloaded data and pretrained weights.
    for folder, manifest in [("data", "download_manifest.json"), ("models", "source.json")]:
        records = json.loads((ROOT / folder / manifest).read_text())
        if isinstance(records, dict):
            records = [records]
        for record in records:
            filename = record.get("file", "mobilenet_v3_small-imagenet.pth")
            path = ROOT / folder / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")

    # 001364457Cheema: Verify image files without training or evaluating predictions.
    dataset = FlowerImages(range(1, 1361), transform=None)
    for content in dataset.images.values():
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    print(f"All {len(dataset)} images are readable.")

    # 001364457Cheema: Check that the pretrained model loads and runs.
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(ROOT / "models/mobilenet_v3_small-imagenet.pth",
                                     map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 1000)
    print("Data, weights and model checks passed.")


if __name__ == "__main__":
    main()
