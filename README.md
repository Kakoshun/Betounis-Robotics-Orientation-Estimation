# Betounis Robotics — Extreme-Point Averaging Orientation Estimation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Research](https://img.shields.io/badge/Research-MDPI%20Electronics-informational) ![License](https://img.shields.io/badge/License-All%20rights%20reserved-lightgrey)

**Standalone Python implementation of the proposed method** for fast estimation of the **undirected orientation axis** of irregularly shaped biological objects using extreme contour points. The implementation does not use neural networks or GPU acceleration.

## Research publication

Ioannis Betounis, Panagiota Germanou, Nikos Petrellis, Christos P. Antonopoulos, and Nikolaos S. Voros. **“Evaluation of Real-Time Orientation Estimation Methods for Various Biological Objects with Irregular Shapes.”** *Electronics* **2026**, 15(21), 4974. https://doi.org/10.3390/electronics15214974

> This repository isolates the proposed orientation estimator. It does **not** contain the five comparison methods, the original datasets, or a complete reproduction of the paper's experiments.

## Algorithm

1. Read an image and convert it to grayscale.
2. Extract edges using `cv2.Canny(gray, 100, 200)`.
3. Select the largest positive-area external contour (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`).
4. Sort contour points by their horizontal (`x`) coordinate.
5. Compute the mean position of the `N` leftmost and `N` rightmost points.
6. Calculate the angle of the vector connecting these two means with `atan2(dy, dx)`.

**Default `N = 2`.** Users can change `N` with `--n`. The algorithm requires at least `2*N` contour points so that the extreme subsets do not overlap. The returned orientation is an **axis**, not a directed heading; angles differing by 180° are equivalent. A 180°-periodic axial error is used for comparisons.

## Requirements

- Python 3.10 or newer
- NumPy and OpenCV (see `requirements.txt`)

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
# Single image, default N=2
python proposed_method.py sample.jpg

# Custom N and output directory
python proposed_method.py sample.jpg --n 5 --output results_n5

# Process a folder, including nested subfolders
python proposed_method.py images --n 2 --output results

# Compare with manually annotated reference angles
python proposed_method.py images --references examples/references_example.csv

# Save CSV without annotated images
python proposed_method.py images --no-annotations
```

### Reference CSV

The optional CSV must contain `Image` and `Reference` columns. `Reference` is an angle in degrees. `Image` may be a filename or a path relative to the input folder (using `/`). Relative paths are recommended if different subfolders contain files with identical names. A basename-only reference is matched only when the basename is unique in the input batch. Missing references are left blank rather than treated as zero.

Example:

```csv
Image,Reference
fish_001.jpg,12.5
leaves/leaf_001.png,-35.0
```

The example is **illustrative**, not a supplied ground-truth dataset.

### Output

`results.csv` contains:

| Column | Meaning |
| --- | --- |
| `Image` | Input image's relative path |
| `N_extreme` | Number of extreme points selected on each side |
| `Estimated_Angle_deg` | Estimated axis angle, or blank on failure |
| `Reference_deg` | Optional reference angle |
| `Axial_Error_deg` | Minimum 180°-periodic angular error, 0–90° |
| `Valid` | `1` if an estimate was obtained, otherwise `0` |
| `Status` | `OK`, `IMAGE_READ_FAILED`, `NO_VALID_CONTOUR`, or `INVALID_ORIENTATION` |
| `Method_Time_us` | Single-call elapsed time of the orientation function in microseconds |

`annotated/` contains visual overlays for valid estimates. Failed images remain in the CSV with an explicit status. The console reports the pooled mean axial error **only over valid images with reference angles**; invalid cases must be considered separately.

## Timing and reproducibility

`Method_Time_us` is a **single Python function-call measurement**, excluding image decoding, Canny, and contour extraction. It is not directly comparable to the paper's **34.28 μs** mean execution time, which used its own benchmarking protocol. The paper reports a **16.17° sample-weighted pooled MAE** and an **18.404° five-dataset macro-average MAE**; this repository cannot independently reproduce these values without the same images, annotations, processing conditions, and aggregation protocol. Results vary with morphology, contour extraction, `N`, and software/hardware.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover geometric cases, axial angle equivalence, input validation, and a synthetic end-to-end image/CSV example. They do not constitute validation on the published datasets.

## Citation

If this implementation supports your work, please cite the associated article. A machine-readable citation is included in `CITATION.cff`.

## Copyright and reuse

**Betounis Robotics — Original Research Implementation**  
**© 2026 Ioannis Betounis. All rights reserved.**

This repository is provided for inspection and research reference. **No open-source license is granted** by the copyright notice. Do not assume that public GitHub visibility permits reuse, modification, or redistribution. Contact the copyright holder for permissions. A separate license can be added later if open-source distribution is intended.
