"""Stage the budget-first Both chart library without touching existing categories."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from optimize_organized_spx_hedges import OUTPUT
from compact_best_spx_images import convert_images, update_browser
from render_organized_spx_charts import (DEFAULT_OUTPUT, Library, load_library, _tasks,
    _render_group, _individual_name, _relative_folder, render_overviews, _draw_overview, write_browser)


def protected_manifest():
    return {str(path.relative_to(DEFAULT_OUTPUT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for category in ("Put buying", "Put selling", "SPX with overlay")
        for path in sorted((DEFAULT_OUTPUT/category).rglob("*")) if path.is_file()}


def run():
    started = time.monotonic()
    manifest = OUTPUT/"preserved_category_sha256.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(protected_manifest(), indent=2), encoding="utf-8")
    library = load_library(OUTPUT)
    library.config["_png_palette_colors"] = 64
    library.config["_allow_webp_resume"] = True
    library.config["_both_webp"] = True
    stage = OUTPUT/"charts"
    stage.mkdir(exist_ok=True)
    subset = Library(library.config, library.metrics[library.metrics.category.eq("Both")].copy(),
        library.dates, library.equity, library.multiple_notionals)
    tasks = _tasks(subset, stage, 125, False, 0)
    tasks = [task for task in tasks if any(
        not (stage/_relative_folder(task[0][0], False)/name).exists()
        and not (stage/_relative_folder(task[0][0], False)/name).with_suffix(".webp").exists()
        for name in [*(_individual_name(row) for row in task[0]), "All expiries.png"])]
    print(f"Rendering {len(tasks)} folders with {len(subset.metrics):,} selected curves", flush=True)
    count = 0
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_render_group, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            count += future.result()["rendered"]
            if i % 25 == 0 or i == len(tasks):
                print(f"Folders {i}/{len(tasks)}; {count:,} PNGs; {time.monotonic()-started:.0f}s", flush=True)
    render_overviews(subset, stage, 125, False)
    _draw_overview(library, library.metrics, stage/"All strategies.png", "All strategy categories",
        "Put selling · Put buying · Combined puts · SPX with overlay", "category", 125, True)
    write_browser(library, stage)
    convert_images()
    update_browser()
    print(f"Staged charts complete: {time.monotonic()-started:.0f}s", flush=True)


if __name__ == "__main__":
    run()
