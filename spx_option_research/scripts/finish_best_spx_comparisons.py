"""Render only missing best-hedge comparisons with bounded plotting memory."""
from optimize_organized_spx_hedges import OUTPUT
import render_organized_spx_charts as renderer
from compact_best_spx_images import update_browser
import argparse
from pathlib import Path
from render_organized_spx_charts import Library, load_library, _draw_overview, _intermediate_groups, TENORS, write_browser


def run(stage=None):
    library = load_library(OUTPUT)
    renderer.MINIMUM_FREE_BYTES = 10*1024*1024
    library.config.update(_png_palette_colors=64, _allow_webp_resume=True, _both_webp=True)
    stage = OUTPUT/"charts" if stage is None else Path(stage)
    frame = library.metrics[library.metrics.category.eq("Both")]
    subset = Library(library.config, frame, library.dates, library.equity, library.multiple_notionals)
    _draw_overview(library, frame, stage/"Both"/"All strategies.png", "Both", "Best historical hedges at all four premium budgets", "tenor", 125, False)
    for i, tenor in enumerate(TENORS):
        selected = frame[frame.tenor.eq(tenor)]
        _draw_overview(library, selected, stage/"Both"/f"{i+1:02d} - {tenor} comparison.png", f"Both · {tenor}", "All short variations and premium budgets", "tenor", 125, False)
    for i, (relative, rows, depth) in enumerate(_intermediate_groups(subset), 1):
        parts = rows.iloc[0].folder.split("/")
        title = parts[0] if depth == 1 else parts[0]+" · Sell "+parts[1].replace("-", "/")
        subtitle = "Both · Best long puts and buffers across short variations and expiries" if depth == 1 else "Both · Best long put and best buffer at each expiry"
        _draw_overview(library, rows, stage/relative/"All strategies.png", title, subtitle, "tenor", 125, False)
        if i % 50 == 0:
            print(f"Intermediate comparisons {i}/424", flush=True)
    _draw_overview(library, library.metrics, stage/"All strategies.png", "All strategy categories",
        "Put selling · Put buying · Combined puts · SPX with overlay", "category", 125, True)
    write_browser(library, stage)
    update_browser(stage)
    print("All comparisons complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args().output_dir)
