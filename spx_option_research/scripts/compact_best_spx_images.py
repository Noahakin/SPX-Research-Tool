"""Encode newly generated Both charts as pixel-identical lossless WebP images."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time

from PIL import Image, ImageChops

OUTPUT = Path(__file__).resolve().parents[1]/"results"/"organized_spx_best_hedges"


def convert(path):
    path = Path(path)
    target = path.with_suffix(".webp")
    temporary = target.with_name("."+target.name+".partial")
    with Image.open(path) as source:
        original = source.convert("RGB")
        original.save(temporary, format="WEBP", lossless=True, method=1)
        with Image.open(temporary) as encoded:
            if encoded.size != original.size or ImageChops.difference(original, encoded.convert("RGB")).getbbox() is not None:
                raise ValueError(f"Image pixels changed: {path}")
    before, after = path.stat().st_size, temporary.stat().st_size
    temporary.replace(target)
    # Only a task-generated PNG is removed, after exact pixel verification.
    path.unlink()
    return before-after


def convert_images():
    root = OUTPUT/"charts"/"Both"
    paths = sorted(root.rglob("*.png"))
    savings = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(convert, path) for path in paths]
        for i, future in enumerate(as_completed(futures), 1):
            savings += future.result()
            if i % 500 == 0 or i == len(paths):
                print(f"Lossless charts {i}/{len(paths)}; saved {savings/1024**2:.1f} MB; {time.monotonic()-started:.0f}s", flush=True)
    return len(paths), savings


def update_browser(root=None):
    path = (Path(root) if root is not None else OUTPUT/"charts")/"index.html"
    text = path.read_text(encoding="utf-8")
    pattern = r'(<script id="library-data" type="application/json">)(.*?)(</script>)'
    match = re.search(pattern, text, re.S)
    rows = json.loads(match.group(2))
    for row in rows:
        if row["category"] == "Both":
            for key in ("path", "expiryPath", "tenorPath", "categoryPath", "basePath", "hedgePath"):
                row[key] = str(Path(row[key]).with_suffix(".webp")).replace("\\", "/")
    payload = json.dumps(rows, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    text = text[:match.start(2)]+payload+text[match.end(2):]
    text = text.replace("Open PNG", "Open image").replace("Save PNG", "Save image")
    text = text.replace("Static PNG charts", "Static charts · Both uses lossless WebP")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    convert_images()
