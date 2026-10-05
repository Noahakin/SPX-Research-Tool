"""Render corrected buffer charts and all affected comparisons in place."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import shutil
import time

from optimize_organized_spx_hedges import OUTPUT
from render_organized_spx_charts import (DEFAULT_OUTPUT, Library, load_library, _tasks,
    _render_group, _draw_overview, _intermediate_groups, TENORS, write_browser)
from compact_best_spx_images import update_browser
import render_organized_spx_charts as renderer


def render_buffer_group(task):
    renderer.MINIMUM_FREE_BYTES = 10*1024*1024
    return _render_group(task)


def run(example=False):
    started = time.monotonic()
    if not example and shutil.disk_usage(DEFAULT_OUTPUT).free < 200*1024*1024:
        raise RuntimeError("Full buffer export needs at least 200 MiB free for remaining images; free 1 GiB to allow working space.")
    library = load_library(OUTPUT)
    renderer.MINIMUM_FREE_BYTES = 10*1024*1024
    library.config.update(_png_palette_colors=64, _allow_webp_resume=True, _both_webp=True)
    frame = library.metrics[library.metrics.category.eq("Both")]
    buffers = frame[frame.hedge_type.eq("Downside buffer")]
    if example:
        buffers = buffers[buffers.short_variation.eq("98-95") & buffers.premium_fraction.eq(.20)]
        renderer.MINIMUM_FREE_BYTES = 1024*1024
    subset = Library(library.config, buffers, library.dates, library.equity, library.multiple_notionals)
    tasks = _tasks(subset, DEFAULT_OUTPUT, 125, False, 0)
    # Keep the user-facing browser and method labels current during the export.
    write_browser(library, DEFAULT_OUTPUT)
    update_browser(DEFAULT_OUTPUT)
    if example:
        for task in tasks:
            _render_group(task)
        _draw_overview(library, frame[frame.short_variation.eq("98-95") & frame.premium_fraction.eq(.20)],
            DEFAULT_OUTPUT/"Both"/"20 percent premium"/"98-95"/"All strategies.png",
            "20 percent premium · Sell 98/95", "Long puts and closest affordable buffers", "tenor", 125, True)
        _draw_overview(library, frame[frame.premium_fraction.eq(.20)],
            DEFAULT_OUTPUT/"Both"/"20 percent premium"/"All strategies.png",
            "20 percent premium", "All short variations · Long puts and closest affordable buffers", "tenor", 125, True)
        _draw_overview(library, frame, DEFAULT_OUTPUT/"Both"/"All strategies.png", "Both",
            "Four premium budgets · Long puts and closest affordable buffers", "tenor", 125, True)
        _draw_overview(library, library.metrics, DEFAULT_OUTPUT/"All strategies.png", "All strategy categories",
            "Put selling · Put buying · Combined puts · SPX with overlay", "category", 125, True)
        print("Rendered 20% premium / 98-95 buffer example, all seven expirations", flush=True)
        return
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(render_buffer_group, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 20 == 0 or i == len(tasks):
                print(f"Buffer folders {i}/{len(tasks)}; {time.monotonic()-started:.0f}s", flush=True)
    both = Library(library.config, frame, library.dates, library.equity, library.multiple_notionals)
    _draw_overview(library, frame, DEFAULT_OUTPUT/"Both"/"All strategies.png", "Both",
        "Four premium budgets · Long puts and closest affordable buffers", "tenor", 125, False)
    for i, tenor in enumerate(TENORS):
        _draw_overview(library, frame[frame.tenor.eq(tenor)], DEFAULT_OUTPUT/"Both"/f"{i+1:02d} - {tenor} comparison.png",
            f"Both · {tenor}", "All short variations and premium budgets", "tenor", 125, False)
    for i, (relative, rows, depth) in enumerate(_intermediate_groups(both), 1):
        parts = rows.iloc[0].folder.split("/")
        title = parts[0] if depth == 1 else parts[0]+" · Sell "+parts[1].replace("-", "/")
        subtitle = "Both · Long puts and closest affordable buffers"
        _draw_overview(library, rows, DEFAULT_OUTPUT/relative/"All strategies.png", title, subtitle, "tenor", 125, False)
        if i % 40 == 0 or i == 424:
            print(f"Folder comparisons {i}/424; {time.monotonic()-started:.0f}s", flush=True)
    _draw_overview(library, library.metrics, DEFAULT_OUTPUT/"All strategies.png", "All strategy categories",
        "Put selling · Put buying · Combined puts · SPX with overlay", "category", 125, True)
    write_browser(library, DEFAULT_OUTPUT)
    update_browser(DEFAULT_OUTPUT)
    print("Corrected chart export complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="store_true")
    run(parser.parse_args().example)
