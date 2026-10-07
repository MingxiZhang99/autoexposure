# autoexposure

Main algorithm code is in [ae_perdict.py](ae_perdict.py):
- `p50`: computes the brightness histogram P50.
- `baseline`: estimates the next exposure from current brightness.
- `autoexposure`: adds trend prediction and light-change limiting on top of baseline.

Dataset generation logic is in [generate_dataset.py](generate_dataset.py):
- Requires one reference image: `reference.JPG`.
- Generates frame sequences under multiple lighting scenarios (gradual change, sudden change, flash, flicker, etc.).
- For each frame, applies illumination scaling plus random Gaussian noise, saves images to `dataset/scene`, and writes metadata to `dataset/dataset.csv`.

Evaluation logic is in [evaluate.py](evaluate.py):
- Loads `dataset.csv` and scene images, then builds per-frame histograms.
- Runs closed-loop auto exposure in three modes: `fixed`, `baseline`, and `predictive`.
- Writes logs and metrics to `results/`, and generates `comparison.txt` plus exposure/brightness curves.

