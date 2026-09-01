"""Set the kernelspec metadata of every notebook to the local cs231n env."""
import json
import os

ROOT = r"E:\ScaleLab\CS231n\project\assignments"
KERNEL = {"name": "cs231n", "display_name": "Python 3.11 (cs231n)", "language": "python"}

n = 0
for dirpath, _dirs, files in os.walk(ROOT):
    for name in files:
        if not name.endswith(".ipynb"):
            continue
        path = os.path.join(dirpath, name)
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        nb.setdefault("metadata", {})["kernelspec"] = KERNEL
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        n += 1
print("kernelspec updated for", n, "notebooks")
