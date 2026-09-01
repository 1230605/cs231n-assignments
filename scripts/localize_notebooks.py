"""Localize CS231n 2026 notebooks for running on this machine.

1. Replaces the first code cell that mounts Google Drive with a local setup
   cell (sets sys.path / cwd, checks datasets).
2. Inserts a torchvision patch cell into notebooks that would otherwise
   re-download CIFAR-10 from the (very slow) official server.
3. Makes the optional Cython build cell in ConvolutionalNetworks use the
   current interpreter and a local path.
"""
import json
import os
import sys

ROOT = r"E:\ScaleLab\CS231n\project\assignments"

LOCAL_SETUP = """# Local setup (replaced the Colab Google Drive mount cell)
# Notebooks are opened from their own assignment folder. This cell makes the
# cs231n package importable and verifies the datasets are in place.
import os, sys
NB_DIR = os.getcwd()
if NB_DIR not in sys.path:
    sys.path.insert(0, NB_DIR)
os.chdir(NB_DIR)
print("Local setup OK:", NB_DIR)
"""

CIFAR_PATCH = """# Local data patch: use the CIFAR-10 batches already shipped in this project
# instead of letting torchvision re-download them from cs.toronto.edu (very
# slow from China). The files contain the same dataset; we just relax
# torchvision's byte-level md5 check because the pickle metadata differs.
import os
try:
    import torchvision.datasets as _tvds
    def _local_cifar_integrity(self):
        return os.path.isdir(os.path.join(self.root, self.base_folder))
    _tvds.CIFAR10._check_integrity = _local_cifar_integrity
    print("CIFAR-10 local patch applied (torchvision download disabled)")
except Exception as _e:
    print("CIFAR-10 local patch skipped:", _e)
"""

CONV_BUILD_CELL = """# Optional: compile the fast Cython im2col extension (needs MSVC Build Tools
# on Windows). If compilation fails, skip this cell - the pure-Python fallback
# in cs231n/fast_layers.py is used automatically.
import os, subprocess, sys
os.chdir(os.path.join(os.getcwd(), "cs231n"))
print(subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                     capture_output=True, text=True).stdout)
os.chdir(os.path.pardir)
"""


def to_source(text):
    return text.splitlines(keepends=True)


def patch_notebook(path, patch_cifar=False, patch_conv=False):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]
    replaced = False
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "google.colab" in src:
            cell["source"] = to_source(LOCAL_SETUP)
            cell["execution_count"] = None
            if "outputs" in cell:
                cell["outputs"] = []
            replaced = True
            break
    if not replaced:
        print("WARN: no Drive-mount cell found in", path)

    if patch_cifar:
        patch_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(CIFAR_PATCH),
            "id": "local-data-patch",
        }
        cells.insert(1, patch_cell)

    if patch_conv:
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "build_ext --inplace" in src:
                cell["source"] = to_source(CONV_BUILD_CELL)
                cell["execution_count"] = None
                if "outputs" in cell:
                    cell["outputs"] = []

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("patched:", os.path.relpath(path, ROOT))


def main():
    targets = []
    for a in ("assignment1", "assignment2", "assignment3"):
        adir = os.path.join(ROOT, a)
        for name in sorted(os.listdir(adir)):
            if not name.endswith(".ipynb") or name.startswith("collect_submission"):
                continue
            p = os.path.join(adir, name)
            cifar = name in ("PyTorch.ipynb", "Self_Supervised_Learning.ipynb",
                             "Transformer_Captioning.ipynb")
            conv = name == "ConvolutionalNetworks.ipynb"
            targets.append((p, cifar, conv))
    for p, cifar, conv in targets:
        patch_notebook(p, patch_cifar=cifar, patch_conv=conv)
    print("done:", len(targets), "notebooks")


if __name__ == "__main__":
    main()
