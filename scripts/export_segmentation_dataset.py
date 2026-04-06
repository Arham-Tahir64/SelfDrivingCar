from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # type: ignore
import numpy as np

from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.perception.segmentation_tasks import (
    derive_boundary_targets,
    remap_carla_semantic_to_task,
    remap_cityscapes_to_task,
    semantic_camera_rgb_to_label_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export remapped segmentation labels and boundaries.")
    parser.add_argument("--semantic-dir", required=True, help="Directory of semantic label images.")
    parser.add_argument("--output-dir", required=True, help="Directory to write remapped labels.")
    parser.add_argument(
        "--source",
        choices=["carla", "cityscapes"],
        default="carla",
        help="Semantic label source format.",
    )
    parser.add_argument(
        "--copy-rgb-dir",
        default=None,
        help="Optional RGB image directory to copy into the export for training convenience.",
    )
    return parser.parse_args()


def _iter_images(directory: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(directory.glob(pattern)))
    return sorted({path.resolve() for path in paths})


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), array)


def main() -> int:
    args = parse_args()
    semantic_dir = Path(args.semantic_dir)
    output_dir = ensure_directory(Path(args.output_dir))
    label_dir = ensure_directory(output_dir / "task_labels")
    lane_dir = ensure_directory(output_dir / "lane_boundary")
    curb_dir = ensure_directory(output_dir / "curb_boundary")

    rgb_dir = Path(args.copy_rgb_dir) if args.copy_rgb_dir else None
    rgb_out_dir = ensure_directory(output_dir / "rgb") if rgb_dir else None

    semantic_paths = _iter_images(semantic_dir)
    if not semantic_paths:
        raise SystemExit(f"no semantic images found in {semantic_dir}")

    manifest: list[dict[str, str]] = []
    for semantic_path in semantic_paths:
        image = _read_image(semantic_path)
        if args.source == "carla":
            raw_labels = semantic_camera_rgb_to_label_map(image)
            task_labels = remap_carla_semantic_to_task(raw_labels)
        else:
            raw_labels = image[..., 0] if image.ndim == 3 else image
            task_labels = remap_cityscapes_to_task(raw_labels)

        lane_boundary, curb_boundary = derive_boundary_targets(task_labels)
        stem = semantic_path.stem
        label_path = label_dir / f"{stem}.png"
        lane_path = lane_dir / f"{stem}.png"
        curb_path = curb_dir / f"{stem}.png"

        _write_image(label_path, task_labels.astype(np.uint8))
        _write_image(lane_path, (lane_boundary * 255.0).astype(np.uint8))
        _write_image(curb_path, (curb_boundary * 255.0).astype(np.uint8))

        record: dict[str, str] = {
            "semantic": str(semantic_path),
            "task_label": str(label_path),
            "lane_boundary": str(lane_path),
            "curb_boundary": str(curb_path),
        }

        if rgb_dir and rgb_out_dir is not None:
            rgb_candidate = next((rgb_dir / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".bmp") if (rgb_dir / f"{stem}{suffix}").exists()), None)
            if rgb_candidate is not None:
                rgb_image = _read_image(rgb_candidate)
                rgb_target = rgb_out_dir / f"{stem}.png"
                _write_image(rgb_target, rgb_image)
                record["rgb"] = str(rgb_target)

        manifest.append(record)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(manifest)} segmentation samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
