import csv
from pathlib import Path
import cv2
import numpy as np


REFERENCE_IMAGE = Path(__file__).resolve().parent / "reference.JPG" 
OUTPUT_DIR = "dataset" 
REFERENCE_EXPOSURE_MS = 10.0  # Fixed reference exposure for scene generation
RNG_SEED = 42  

SCENARIOS = [ 
    "constant",
    "gradual_darkening",
    "gradual_brightening",
    "sudden_darkening",
    "sudden_brightening",
    "flash",
    "flicker",
    "dark_bright_dark",
]

def illumination_for(scenario, frame):
    if scenario == "constant":
        return 1.0

    # Slow darkening then recovery
    if scenario == "gradual_darkening":
        start_ratio = min(max((frame - 10) / 30, 0.0), 1.0)
        recover_ratio = min(max((frame - 60) / 30, 0.0), 1.0)
        return 1.0 - 0.65 * (start_ratio - recover_ratio)

    # Slow brightening then recovery
    if scenario == "gradual_brightening":
        start_ratio = min(max((frame - 10) / 30, 0.0), 1.0)
        recover_ratio = min(max((frame - 60) / 30, 0.0), 1.0)
        return 1.0 + 0.65 * (start_ratio - recover_ratio)

    # Fast darkening and fast recovery
    if scenario == "sudden_darkening":
        start_ratio = min(max((frame - 20) / 3, 0.0), 1.0)
        recover_ratio = min(max((frame - 60) / 3, 0.0), 1.0)
        return 1.0 - 0.65 * (start_ratio - recover_ratio)

    # Fast brightening and fast recovery
    if scenario == "sudden_brightening":
        start_ratio = min(max((frame - 20) / 3, 0.0), 1.0)
        recover_ratio = min(max((frame - 60) / 3, 0.0), 1.0)
        return 1.0 + 0.65 * (start_ratio - recover_ratio)

    # Three fixed flashes in bright dark bright order
    if scenario == "flash":
        flash_starts = [16, 47, 78]
        flash_gains = [2.5, 0.4, 2.5]
        for i in range(len(flash_starts)):
            start = flash_starts[i]
            gain = flash_gains[i]
            if start <= frame < start + 3:
                return gain
        return 1.0

    # Sinusoidal flicker for seven full cycles
    if scenario == "flicker":
        if frame < 84:
            return 1.0 + 0.35 * np.sin(2.0 * np.pi * frame / 12)
        return 1.0

    # Piecewise dark bright dark pattern
    if scenario == "dark_bright_dark":
        if frame < 10:
            return 1.0
        if frame < 30:
            return 0.45
        if frame < 60:
            return 1.8
        if frame < 80:
            return 0.55
        return 1.0

    raise ValueError(f"Unknown scenario: {scenario}")


def make_base_scene(reference_bgr, illumination, rng):
    base = reference_bgr.astype(np.float32) * illumination
    noise = rng.normal(0.0, 1.5, size=base.shape).astype(np.float32)
    base = np.minimum(np.maximum(base + noise, 0.0), 255.0)
    return base.astype(np.uint8)


def main():
    output_dir = Path(__file__).resolve().parent / OUTPUT_DIR
    scene_dir = output_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    reference = cv2.imread(str(REFERENCE_IMAGE), cv2.IMREAD_COLOR)
    rng = np.random.default_rng(RNG_SEED)

    csv_path = output_dir / "dataset.csv"

    frame_number = 1

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", "scenario", "local_frame", "illumination", "filename"]
        )

        for scenario in SCENARIOS:
            for local_frame in range(100):  
                illumination = illumination_for(scenario, local_frame)
                image = make_base_scene(reference, illumination, rng)

                filename = f"frame_{frame_number:04d}.jpg"
                cv2.imwrite(str(scene_dir / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

                writer.writerow(
                    [
                        frame_number,
                        scenario,
                        local_frame,
                        f"{illumination:.8f}",
                        filename,
                    ]
                )

                frame_number += 1


if __name__ == "__main__":
    main()
