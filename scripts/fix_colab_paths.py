"""Replace remaining Drive-absolute paths in DDPM / CLIP_DINO with local ones."""
import json
import os

BASE = r"E:\ScaleLab\CS231n\project\assignments\assignment3"

REPLACEMENTS = {
    "DDPM.ipynb": [
        ("Image(f'/content/drive/My Drive/{FOLDERNAME}/unet.png')",
         "Image('unet.png')"),
        ('results_folder = f"/content/drive/My Drive/{FOLDERNAME}/cs231n/exp/pretrained"',
         'results_folder = "cs231n/exp/pretrained"'),
    ],
    "CLIP_DINO.ipynb": [
        ("ColabImage(f'/content/drive/My Drive/{FOLDERNAME}/CLIP.png')",
         "ColabImage('CLIP.png')"),
        ("ColabImage(f'/content/drive/My Drive/{FOLDERNAME}/dino.gif')",
         "ColabImage('dino.gif')"),
        ('write_video_from_array(final_viz, f"/content/drive/My Drive/{FOLDERNAME}/dino_res.mp4")',
         'write_video_from_array(final_viz, "dino_res.mp4")'),
    ],
}

for fname, pairs in REPLACEMENTS.items():
    path = os.path.join(BASE, fname)
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    count = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = src
        for old, new in pairs:
            if old in new_src:
                new_src = new_src.replace(old, new)
                count += 1
        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
            cell["execution_count"] = None
            if "outputs" in cell:
                cell["outputs"] = []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(fname, "->", count, "replacements")
