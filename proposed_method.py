# Betounis Robotics | Original Research Implementation | © 2026 Ioannis Betounis
# Copyright (c) 2026 Ioannis Betounis. All rights reserved.
# Publication: https://doi.org/10.3390/electronics15214974
"""Extreme-point averaging orientation estimation for undirected 2D object axes."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
RESULT_FIELDS = ['Image', 'N_extreme', 'Estimated_Angle_deg', 'Reference_deg',
                 'Axial_Error_deg', 'Valid', 'Status', 'Method_Time_us']


def normalize_angle(angle: float) -> float:
    """Normalize a directed angle to (-180, 180] degrees."""
    result = float(angle) % 360.0
    return result - 360.0 if result > 180.0 else result


def axial_error(estimate: float | None, reference: float | None) -> float | None:
    """Return the smallest difference between undirected axes (0 to 90 degrees)."""
    if estimate is None or reference is None:
        return None
    if not math.isfinite(estimate) or not math.isfinite(reference):
        return None
    delta = abs(float(estimate) - float(reference)) % 180.0
    return min(delta, 180.0 - delta)


def proposed_orientation(contour: np.ndarray | None, n_extreme: int = 2):
    """Return (angle, left_mean, right_mean) using N left/right extreme points.

    Points are sorted by x, using a stable sort to preserve the original
    ordering among equal-x points. The result describes an axis, not heading.
    An invalid contour returns (None, None, None). At least 2*N points are
    required so the selected extreme subsets do not overlap.
    """
    if isinstance(n_extreme, bool) or not isinstance(n_extreme, (int, np.integer)) or n_extreme < 1:
        raise ValueError('n_extreme must be a positive integer')
    if contour is None:
        return None, None, None
    points = np.asarray(contour)
    if points.size == 0 or points.size % 2:
        return None, None, None
    try:
        points = points.reshape(-1, 2)
        if not np.isfinite(points).all() or len(points) < 2 * n_extreme:
            return None, None, None
        ordered = sorted(points, key=lambda point: point[0])
        left = np.mean(ordered[:n_extreme], axis=0, dtype=np.float64)
        right = np.mean(ordered[-n_extreme:], axis=0, dtype=np.float64)
        left_xy = tuple(map(float, left))
        right_xy = tuple(map(float, right))
        dx, dy = (right - left).tolist()
        if dx == 0 and dy == 0:
            return None, left_xy, right_xy
        return normalize_angle(math.degrees(math.atan2(dy, dx))), left_xy, right_xy
    except (TypeError, ValueError, OverflowError):
        return None, None, None


def extract_contour(image: np.ndarray | None) -> np.ndarray | None:
    """Extract the largest positive-area external Canny contour."""
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        return None
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour if cv2.contourArea(contour) > 0 else None


def annotate(image, contour, left, right, angle, n_extreme):
    """Overlay the selected contour, extreme-point means and orientation axis."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(out, [contour], -1, (0, 200, 0), 1)
    p1 = tuple(int(round(v)) for v in left)
    p2 = tuple(int(round(v)) for v in right)
    cv2.line(out, p1, p2, (0, 0, 255), 2)
    cv2.circle(out, p1, 4, (255, 0, 0), -1)
    cv2.circle(out, p2, 4, (255, 0, 0), -1)
    cv2.putText(out, f'Axis: {angle:.2f} deg | N={n_extreme}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def load_references(path: Path) -> dict[str, float]:
    """Read Image/Reference pairs; reject ambiguous or duplicate identifiers."""
    references = {}
    with path.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'Image', 'Reference'}.issubset(reader.fieldnames or []):
            raise ValueError('Reference CSV must contain Image and Reference columns')
        for line, row in enumerate(reader, start=2):
            name = (row.get('Image') or '').strip().replace('\\', '/')
            raw = (row.get('Reference') or '').strip()
            if not name or not raw:
                continue
            try:
                angle = float(raw)
            except ValueError as exc:
                raise ValueError(f'Invalid reference angle at CSV line {line}') from exc
            if not math.isfinite(angle):
                raise ValueError(f'Non-finite reference angle at CSV line {line}')
            if name in references:
                raise ValueError(f'Duplicate reference Image: {name}')
            references[name] = angle
    return references


def run(input_path: Path, output: Path, n_extreme: int = 2,
        references_path: Path | None = None, save_annotations: bool = True):
    """Process images and save per-image CSV results; return result rows."""
    if not input_path.exists():
        raise FileNotFoundError(f'Input not found: {input_path}')
    if n_extreme < 1:
        raise ValueError('N must be at least 1')
    if input_path.is_dir():
        images = sorted(p for p in input_path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        images = [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not images:
        raise ValueError('No supported image files found')
    refs = load_references(references_path) if references_path else {}
    # Basename-only matches are safe only when unique within the input batch.
    basename_counts = {}
    for image_path in images:
        basename_counts[image_path.name] = basename_counts.get(image_path.name, 0) + 1
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in images:
        relative = path.relative_to(input_path) if input_path.is_dir() else Path(path.name)
        key = relative.as_posix()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        status = 'OK'
        contour = None
        if image is None:
            status = 'IMAGE_READ_FAILED'
        else:
            contour = extract_contour(image)
            if contour is None:
                status = 'NO_VALID_CONTOUR'
        start = perf_counter_ns()
        angle, left, right = proposed_orientation(contour, n_extreme)
        elapsed_us = (perf_counter_ns() - start) / 1000.0
        if status == 'OK' and angle is None:
            status = 'INVALID_ORIENTATION'
        reference = refs.get(key)
        if reference is None and basename_counts[path.name] == 1:
            reference = refs.get(path.name)
        error = axial_error(angle, reference)
        rows.append({
            'Image': key, 'N_extreme': n_extreme,
            'Estimated_Angle_deg': '' if angle is None else f'{angle:.10f}',
            'Reference_deg': '' if reference is None else f'{reference:.10f}',
            'Axial_Error_deg': '' if error is None else f'{error:.10f}',
            'Valid': int(angle is not None), 'Status': status,
            'Method_Time_us': f'{elapsed_us:.3f}'
        })
        if angle is not None and save_annotations:
            target = output / 'annotated' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), annotate(image, contour, left, right, angle, n_extreme)):
                raise OSError(f'Could not save annotation: {target}')
        print(f'{key}: {status}' + (f' | {angle:.3f} deg' if angle is not None else ''))
    with (output / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    valid_errors = [float(row['Axial_Error_deg']) for row in rows if row['Axial_Error_deg'] != '']
    print(f'Images: {len(rows)} | Valid: {sum(r["Valid"] for r in rows)} | Invalid: {sum(not r["Valid"] for r in rows)}')
    if valid_errors:
        print(f'Pooled axial MAE (valid, reference-matched): {np.mean(valid_errors):.4f} deg (n={len(valid_errors)})')
    print(f'Results: {output / "results.csv"}')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Betounis Robotics | Proposed extreme-point averaging method')
    parser.add_argument('input', type=Path, help='Image file or folder')
    parser.add_argument('--n', type=int, default=2, help='Extreme points per side; default 2')
    parser.add_argument('--output', type=Path, default=Path('results'), help='Output directory')
    parser.add_argument('--references', type=Path, help='Optional CSV with Image and Reference columns')
    parser.add_argument('--no-annotations', action='store_true', help='Do not save annotated images')
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.n, args.references, not args.no_annotations)
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
