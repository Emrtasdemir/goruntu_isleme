# -*- coding: utf-8 -*-
# export_fashionmnist_to_png.py
# FashionMNIST'i PNG + labels.txt yapısına çevirir.

from pathlib import Path
from PIL import Image
from torchvision import datasets
import os


LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def export_split(dataset, split_dir: Path):
    """Verilen dataset'i split_dir içine PNG + labels.txt olarak kaydeder."""
    split_dir.mkdir(parents=True, exist_ok=True)

    labels_file = split_dir / "labels.txt"
    with labels_file.open("w", encoding="utf-8") as f:
        for idx, (img, target) in enumerate(dataset):
            # torchvision FashionMNIST: img zaten PIL.Image
            img = img.convert("L")  # Gri seviye
            fname = f"{idx:06d}.png"
            img.save(split_dir / fname, "PNG")
            f.write(f"{fname} {int(target)}\n")


def main():
    # Script'in bulunduğu klasör
    root_dir = Path(__file__).parent

    # Çıkış kök klasörü: acikhali2
    out_root = (root_dir / "acikhali2").resolve()
    train_dir = out_root / "train"
    test_dir = out_root / "test"

    # Torch cache klasörü (indirilen orijinal dataset için)
    cache = out_root / "_torch_cache"
    cache.mkdir(parents=True, exist_ok=True)

    # FashionMNIST dataset'ini indir
    print("[INFO] FashionMNIST indiriliyor / yükleniyor...")
    train_ds = datasets.FashionMNIST(
        root=str(cache), train=True, download=True
    )
    test_ds = datasets.FashionMNIST(
        root=str(cache), train=False, download=True
    )

    print(f"[INFO] Exporting TRAIN -> {train_dir}")
    export_split(train_ds, train_dir)

    print(f"[INFO] Exporting TEST  -> {test_dir}")
    export_split(test_ds, test_dir)

    print("[OK] Done.")
    print(f"[OK] labels 0..9: {LABELS}")


if __name__ == "__main__":
    main()

