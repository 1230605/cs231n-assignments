"""Reconstruct CIFAR-10 batches-py format from HuggingFace uoft-cs/cifar10 parquet files.

The PNG-encoded images are lossless, so the reconstructed pickle batches are
byte-identical in pixel values to the official cifar-10-python.tar.gz.
"""
import io
import os
import pickle
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

DATA_DIR = r"E:\ScaleLab\CS231n\project\data"
OUT_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]


def load_images(parquet_path):
    table = pq.read_table(parquet_path)
    imgs = table.column("img").to_pylist()
    labels = table.column("label").to_pylist()
    data = np.empty((len(imgs), 32 * 32 * 3), dtype=np.uint8)
    for i, item in enumerate(imgs):
        img = Image.open(io.BytesIO(item["bytes"]))
        data[i] = np.asarray(img, dtype=np.uint8).reshape(-1)
    return data, labels


def write_batch(path, data, labels, batch_label):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "batch_label": batch_label,
                "labels": list(labels),
                "data": data,
                "num_cases_per_batch": len(data),
            },
            f,
            protocol=2,
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    Xtr, ytr = load_images(os.path.join(DATA_DIR, "train-00000-of-00001.parquet"))
    Xte, yte = load_images(os.path.join(DATA_DIR, "test-00000-of-00001.parquet"))
    assert Xtr.shape[0] == 50000 and Xte.shape[0] == 10000

    for b in range(5):
        start, end = b * 10000, (b + 1) * 10000
        write_batch(
            os.path.join(OUT_DIR, "data_batch_%d" % (b + 1)),
            Xtr[start:end], ytr[start:end], "training batch %d of 5" % (b + 1),
        )
    write_batch(os.path.join(OUT_DIR, "test_batch"), Xte, yte, "testing batch 1 of 1")

    with open(os.path.join(OUT_DIR, "batches.meta"), "wb") as f:
        pickle.dump(
            {
                "num_cases_per_batch": 10000,
                "label_names": LABELS,
                "num_vis": 3072,
            },
            f,
            protocol=2,
        )

    # Sanity checks
    for name in ["data_batch_1", "data_batch_5", "test_batch"]:
        with open(os.path.join(OUT_DIR, name), "rb") as f:
            d = pickle.load(f, encoding="latin1")
            arr = np.asarray(d["data"]).reshape(-1, 3, 32, 32)
            print(name, "->", arr.shape, "labels:", len(d["labels"]),
                  "| first label:", d["labels"][0], "| pixel range:",
                  int(arr.min()), int(arr.max()))


if __name__ == "__main__":
    main()
