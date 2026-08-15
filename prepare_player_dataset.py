"""Convert LabelMe / X-AnyLabeling JSON to YOLO dataset with train/val/test split."""

import json
import random
import shutil
from pathlib import Path
from typing import Optional

SOURCE_FOLDERS = [
    Path(r"D:\视频\frames\merged_1s"),
    Path(r"D:\视频\Captures\frames\2026-07-29 13-19-54"),
    Path(r"D:\视频\Captures\frames\2026-07-29 11-54-39"),
    Path(r"D:\视频\frames\7月29日"),
]

OUTPUT_ROOT = Path(r"C:\Users\35882\Desktop\hypn\player_dataset")
CLASS_NAME = "Player"
CLASS_ID = 0
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def labelme_to_yolo_line(shape: dict, img_w: int, img_h: int) -> Optional[str]:
    if shape.get("shape_type") != "rectangle":
        return None
    if shape.get("label") != CLASS_NAME:
        return None

    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)
    if width <= 0 or height <= 0:
        return None

    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def collect_samples() -> list[tuple[Path, list[str]]]:
    samples: list[tuple[Path, list[str]]] = []
    for folder in SOURCE_FOLDERS:
        if not folder.exists():
            continue
        for json_path in sorted(folder.glob("*.json")):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            img_name = data.get("imagePath") or (json_path.stem + ".jpg")
            img_path = folder / img_name
            if not img_path.exists():
                continue

            img_w = data["imageWidth"]
            img_h = data["imageHeight"]
            lines = []
            for shape in data.get("shapes", []):
                line = labelme_to_yolo_line(shape, img_w, img_h)
                if line:
                    lines.append(line)

            if lines:
                samples.append((img_path, lines))
    return samples


def write_data_yaml(root: Path, dataset_path: str | None = None) -> None:
    base = dataset_path or "."
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {base}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                f"  {CLASS_ID}: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def split_samples(samples: list) -> tuple[list, list, list]:
    random.shuffle(samples)
    n = len(samples)
    val_count = max(1, round(n * VAL_RATIO))
    test_count = max(1, round(n * TEST_RATIO))
    if val_count + test_count >= n:
        val_count = max(1, n // 5)
        test_count = max(1, n // 5)
    val_samples = samples[:val_count]
    test_samples = samples[val_count : val_count + test_count]
    train_samples = samples[val_count + test_count :]
    if not train_samples:
        train_samples = samples[val_count + test_count - 1 : val_count + test_count]
        test_samples = samples[val_count : val_count + test_count - 1]
    return train_samples, val_samples, test_samples


def reset_output(root: Path) -> None:
    if root.exists():
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                d = root / sub / split
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    random.seed(SEED)
    samples = collect_samples()
    if not samples:
        raise SystemExit("No labeled samples found.")

    reset_output(OUTPUT_ROOT)

    train_samples, val_samples, test_samples = split_samples(samples)

    for split, subset in (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ):
        for idx, (img_path, lines) in enumerate(subset):
            stem = f"{img_path.parent.name}_{img_path.stem}_{idx:04d}"
            dst_img = OUTPUT_ROOT / "images" / split / f"{stem}.jpg"
            dst_lbl = OUTPUT_ROOT / "labels" / split / f"{stem}.txt"
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_data_yaml(OUTPUT_ROOT)

    total_boxes = sum(len(lines) for _, lines in samples)
    print(f"Samples: {len(samples)} (train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)})")
    print(f"Total Player boxes: {total_boxes}")
    print(f"Dataset: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
