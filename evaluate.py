import csv
import time
from pathlib import Path
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from ae_perdict import autoexposure, baseline, p50

matplotlib.use("Agg")

REFERENCE_EXPOSURE_MS = 10.0
TARGET_BRIGHTNESS = 128.0
MIN_EXPOSURE_MS = 1.0
MAX_EXPOSURE_MS = 40.0
PRED_CONFIDENCE_THRESHOLD = 0.6
PRED_MAX_LIGHT_CHANGE_RATIO = 0.6


def controller_for_mode(mode, target_brightness, min_exposure_ms, max_exposure_ms):
    if mode == "fixed":
        return lambda hist, exposure_ms: min(max(exposure_ms, min_exposure_ms), max_exposure_ms)
    if mode == "baseline":
        return lambda hist, exposure_ms: baseline(
            hist,
            exposure_ms,
            target_brightness,
            min_exposure_ms,
            max_exposure_ms,
        )
    if mode == "predictive":
        return autoexposure(
            target_brightness,
            min_exposure_ms,
            max_exposure_ms,
            confidence_threshold=PRED_CONFIDENCE_THRESHOLD,
            max_light_change_ratio=PRED_MAX_LIGHT_CHANGE_RATIO,
        )
    raise ValueError(f"Unknown mode: {mode}")


def load_frame_rows(dataset_dir):
    with (dataset_dir / "dataset.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_base_hists(dataset_dir, frame_rows):
    base_hists = []
    for frame_meta in frame_rows:
        gray = cv2.imread(str(dataset_dir / "scene" / frame_meta["filename"]), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if gray is None:
            raise FileNotFoundError(frame_meta["filename"])
        base_hists.append(np.bincount(gray.ravel(), minlength=256))
    return base_hists


def compute_perfect_exposure(frame_rows, reference_image_path):
    ref = cv2.imread(str(reference_image_path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if ref is None:
        raise FileNotFoundError(reference_image_path)
    ref_p50 = p50(np.bincount(ref.ravel(), minlength=256))
    light = np.array([float(r["illumination"]) for r in frame_rows])
    perfect = REFERENCE_EXPOSURE_MS * TARGET_BRIGHTNESS / (ref_p50 * light)
    return perfect, ref_p50


def run_closed_loop(
    frame_rows,
    base_hists,
    mode,
    initial_exposure_ms=10.0,
    target_brightness=128.0,
    min_exposure_ms=1.0,
    max_exposure_ms=40.0,
):
    controller = controller_for_mode(mode, target_brightness, min_exposure_ms, max_exposure_ms)
    exposure = min(max(initial_exposure_ms, min_exposure_ms), max_exposure_ms)
    levels = np.arange(256)
    records = []
    for i in range(len(frame_rows)):
        frame_meta = frame_rows[i]
        base_hist = base_hists[i]
        new_level = np.minimum(np.round(levels * exposure / REFERENCE_EXPOSURE_MS), 255).astype(int)
        hist = np.bincount(new_level, base_hist, 256)

        t0 = time.perf_counter()
        next_exposure = controller(hist, exposure)
        ctrl_us = (time.perf_counter() - t0) * 1e6

        records.append(
            {
                "frame_id": int(frame_meta["frame"]),
                "image_name": frame_meta["filename"],
                "current_exposure_ms": exposure,
                "brightness_p50": p50(hist),
                "next_exposure_ms": next_exposure,
                "ctrl_time_us": ctrl_us,
            }
        )
        exposure = min(max(next_exposure, min_exposure_ms), max_exposure_ms)
    return records


def write_records_csv(records, csv_path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()} for r in records)


def evaluate(frame_rows, records, perfect, target_brightness):
    exposure = np.array([r["current_exposure_ms"] for r in records])
    p50_values = np.array([r["brightness_p50"] for r in records])
    ctrl_us = np.array([r["ctrl_time_us"] for r in records])

    summary = []
    for scenario in dict.fromkeys(r["scenario"] for r in frame_rows):
        mask = np.array([r["scenario"] == scenario for r in frame_rows])
        summary.append(
            {
                "scenario": scenario,
                "exposure_mae": float(np.abs(exposure[mask] - perfect[mask]).mean()),
                "p50_err": float(np.abs(p50_values[mask] - target_brightness).mean()),
                "ctrl_us": float(ctrl_us[mask].mean()),
            }
        )
    return summary


def write_comparison_table(table_path, dataset_name, frame_count, ref_p50, modes, summaries):
    notes = {
        "exposure_mae": "Mean absolute error to perfect exposure (ms), lower is better",
        "p50_err": "Mean absolute P50 error from target 128, lower is better",
        "ctrl_us": "Mean controller compute time per frame (μs), lower is better",
    }

    lines = [f"Dataset {dataset_name} ({frame_count} frames), reference P50={ref_p50:.1f}", ""]
    for key, note in notes.items():
        lines.append(f"[{key}] {note}")
        lines.append(f"{'scenario':<22}" + "".join(f"{mode:>12}" for mode in modes))
        for i, row in enumerate(summaries[modes[0]]):
            values = [summaries[mode][i][key] for mode in modes]
            lines.append(f"{row['scenario']:<22}" + "".join(f"{v:>12.4g}" for v in values))
        lines.append("")

    table_path.write_text("\n".join(lines), encoding="utf-8")


def plot_curves(frame_rows, results, perfect, output_png):
    scenarios = list(dict.fromkeys(r["scenario"] for r in frame_rows))
    fig, axes = plt.subplots(2, len(scenarios), figsize=(3 * len(scenarios), 6))

    for j, scenario in enumerate(scenarios):
        mask = np.array([r["scenario"] == scenario for r in frame_rows])
        axes[0, j].plot(perfect[mask], "k--", label="perfect")
        axes[1, j].axhline(TARGET_BRIGHTNESS, color="k", ls="--")
        for mode, records in results.items():
            axes[0, j].plot(np.array([r["current_exposure_ms"] for r in records])[mask], label=mode)
            axes[1, j].plot(np.array([r["brightness_p50"] for r in records])[mask], label=mode)
        axes[0, j].set_title(scenario, fontsize=9)

    axes[0, 0].set_ylabel("exposure (ms)")
    axes[1, 0].set_ylabel("P50")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_png, dpi=110)
    plt.close(fig)


def main():
    here = Path(__file__).parent
    dataset_dir = here / "dataset"
    reference_image = here / "reference.JPG"
    results_dir = here / "results"
    initial_exposure_ms = 10.0

    frame_rows = load_frame_rows(dataset_dir)
    base_hists = load_base_hists(dataset_dir, frame_rows)
    perfect, ref_p50 = compute_perfect_exposure(frame_rows, reference_image)

    modes = ["fixed", "baseline", "predictive"]
    results = {}
    summaries = {}
    for mode in modes:
        records = run_closed_loop(
            frame_rows,
            base_hists,
            mode,
            initial_exposure_ms=initial_exposure_ms,
            target_brightness=TARGET_BRIGHTNESS,
            min_exposure_ms=MIN_EXPOSURE_MS,
            max_exposure_ms=MAX_EXPOSURE_MS,
        )
        results[mode] = records
        summaries[mode] = evaluate(frame_rows, records, perfect, TARGET_BRIGHTNESS)

        out = results_dir / f"results_{mode}"
        out.mkdir(parents=True, exist_ok=True)
        write_records_csv(records, out / f"{mode}_log.csv")

    results_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(results["predictive"], results_dir / "predictive_log.csv")

    table_path = results_dir / "comparison.txt"
    write_comparison_table(table_path, dataset_dir.name, len(frame_rows), ref_p50, modes, summaries)
    print(f"Comparison table: {table_path}")

    curve_path = results_dir / "ae_curves.png"
    plot_curves(frame_rows, results, perfect, curve_path)
    print(f"Curve plot: {curve_path}")


if __name__ == "__main__":
    main()
