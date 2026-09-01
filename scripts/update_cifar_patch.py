"""Update the 'local-data-patch' cell in notebooks with an enhanced torchvision patch."""
import json
import os

BASE = r"E:\ScaleLab\CS231n\project\assignments"
TARGETS = [
    os.path.join(BASE, "assignment2", "PyTorch.ipynb"),
    os.path.join(BASE, "assignment3", "Self_Supervised_Learning.ipynb"),
    os.path.join(BASE, "assignment3", "Transformer_Captioning.ipynb"),
]

PATCH = """# Local data patch: use the CIFAR-10 batches already shipped in this project
# instead of letting torchvision re-download them from cs.toronto.edu (very
# slow from China). The files contain the same dataset; we relax torchvision's
# byte-level md5 checks because the pickle metadata differs.
import os, pickle
try:
    import torchvision.datasets as _tvds
    def _local_cifar_integrity(self):
        return os.path.isdir(os.path.join(self.root, self.base_folder))
    def _local_load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        self.classes = meta["label_names"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
    _tvds.CIFAR10._check_integrity = _local_cifar_integrity
    _tvds.CIFAR10._load_meta = _local_load_meta
    print("CIFAR-10 local patch applied (torchvision download disabled)")
except Exception as _e:
    print("CIFAR-10 local patch skipped:", _e)
"""

for path in TARGETS:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    n = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and cell.get("id") == "local-data-patch":
            cell["source"] = PATCH.splitlines(keepends=True)
            cell["execution_count"] = None
            if "outputs" in cell:
                cell["outputs"] = []
            n += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(os.path.basename(path), "->", n, "patch cell(s) updated")
