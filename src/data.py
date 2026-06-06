"""
Dataset loading and splitting helpers

This module reads JSON annotations, maps the original tags to two project
classes, makes one reproducible train/test split, and validates image paths
"""
from dataclasses import dataclass
import json
from pathlib import Path

from src.config import DATA_DIR, PROJECT_ROOT, TEST_RATIO

# The annotation file uses `is_empty` and `is_full`. The API exposes collection
# labels, so the code maps them once here
CLASS_NAMES = ["has_space", "needs_collection"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class ImageRecord:
    """Annotated image path and integer class label"""

    image_path: Path
    label: int


def resolve_data_dir(data_dir=None):
    """Resolve a dataset path from CLI input, cwd, or project root

    Args:
        data_dir: Dataset path from the CLI or None to use DATA_DIR

    Returns:
        Absolute dataset path
    """

    if data_dir is None:
        return DATA_DIR

    path = Path(data_dir)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def _label_from_tags(tags):
    """Map one annotation tag set to the project class

    Args:
        tags: Annotation labels attached to one image

    Returns:
        Integer class label for the classifier

    Raises:
        ValueError: If tags are conflicting or missing a fill-level label
    """

    if "is_full" in tags and "is_empty" in tags:
        raise ValueError(f"Conflicting labels: {tags}")
    if "is_full" in tags:
        return LABEL_MAP["needs_collection"]
    if "is_empty" in tags:
        return LABEL_MAP["has_space"]
    raise ValueError(f"Missing fill-level label: {tags}")


def _image_number(image_path):
    """Return the numeric image id used to order the split

    Args:
        image_path: Image file path whose stem may contain digits

    Returns:
        Extracted image id, or 0 when the stem has no digits
    """

    stem = Path(image_path).stem
    digits = "".join(char for char in stem if char.isdigit())
    if not digits:
        return 0
    return int(digits)


def load_annotations(data_dir=None):
    """Read annotations.json and validate image records

    Missing labels or image files raise errors before training starts

    Args:
        data_dir: Dataset directory or None to use DATA_DIR

    Returns:
        Image records, class names, and class counts

    Raises:
        FileNotFoundError: If annotations or referenced images are missing
        ValueError: If annotations are empty or contain invalid labels
    """

    data_path = resolve_data_dir(data_dir)
    annotations_path = data_path / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")

    records = []
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))

    for image_data in annotations.get("images", []):
        name = image_data.get("name")
        if not name:
            raise ValueError("Annotation image without name")

        tags = tuple(sorted(image_data.get("labels", [])))
        label = _label_from_tags(tags)
        image_path = data_path / name
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        records.append(ImageRecord(
            image_path=image_path,
            label=label,
        ))

    if not records:
        raise ValueError(f"No image annotations found in {annotations_path}")

    return records, CLASS_NAMES, _count_records(records)


def split_records(records, test_ratio=TEST_RATIO):
    """Create stratified train and test records

    Each class contributes the same percentage of images to the test set. After
    sorting by image id, the highest-numbered images become test records. This
    keeps repeated runs on the same files without storing a split file

    Args:
        records: Annotated image records to split
        test_ratio: Fraction of each class to place in the test split

    Returns:
        Train records, test records, and split count metadata
    """

    by_class = {idx: [] for idx in range(len(CLASS_NAMES))}
    # The report uses one reproducible stratified holdout. Sorting by image
    # number keeps the same train/test files on every run
    for record in sorted(records, key=lambda item: _image_number(item.image_path)):
        by_class[record.label].append(record)

    test_record_set = set()
    test_class_counts = {}
    for class_idx, class_records in by_class.items():
        class_name = CLASS_NAMES[class_idx]
        requested_count = int(len(class_records) * test_ratio)
        if requested_count == 0 and len(class_records) > 1:
            requested_count = 1
        test_class_counts[class_name] = requested_count
        if requested_count > 0:
            test_record_set.update(class_records[-requested_count:])

    test_records = [record for record in records if record in test_record_set]
    train_records = [record for record in records if record not in test_record_set]
    split_info = {
        "train": _count_records(train_records),
        "test": _count_records(test_records),
        "test_class_counts": test_class_counts,
        "test_strategy": "stratified_holdout",
    }
    return train_records, test_records, split_info


def _count_records(records):
    """Count records per class for reporting and console output

    Args:
        records: Annotated image records to count

    Returns:
        Mapping from class name to record count
    """

    counts = {name: 0 for name in CLASS_NAMES}
    for record in records:
        counts[CLASS_NAMES[record.label]] += 1
    return counts
