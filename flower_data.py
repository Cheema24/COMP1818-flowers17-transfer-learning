import io
import tarfile
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset

ROOT = Path(__file__).resolve().parent


class FlowerImages(Dataset):

    def __init__(self, image_ids, transform, archive_path=None):
        self.image_ids = [int(i) for i in image_ids]
        if not self.image_ids or any(i < 1 or i > 1360 for i in self.image_ids):
            raise ValueError("Expected Flowers-17 image IDs in the range 1..1360.")
        self.transform = transform
        wanted = set(self.image_ids)
        self.images = {}
        archive_path = Path(archive_path) if archive_path is not None else ROOT / "data" / "17flowers.tgz"
        # 001364457Cheema: Read the archive once to avoid cloud file stalls.
        blob = archive_path.read_bytes()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r|gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or not name.startswith("image_") or not name.endswith(".jpg"):
                    continue
                try:
                    image_id = int(name[6:-4])
                except ValueError:
                    continue
                if image_id in wanted:
                    if image_id in self.images:
                        raise ValueError(f"Duplicate image ID in archive: {image_id}")
                    with archive.extractfile(member) as file:
                        self.images[image_id] = file.read()
        missing = wanted - self.images.keys()
        if missing:
            raise ValueError(f"Archive is missing requested image IDs: {sorted(missing)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        with Image.open(io.BytesIO(self.images[image_id])) as image:
            image = self.transform(image.convert("RGB"))
        return image, (image_id - 1) // 80


def load_tensors(image_ids, transform):
    # 001364457Cheema: Prepare images once for repeated training epochs.
    dataset = FlowerImages(image_ids, transform)
    images, labels = zip(*(dataset[i] for i in range(len(dataset))))
    return TensorDataset(torch.stack(images), torch.tensor(labels))
