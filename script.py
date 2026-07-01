import argparse
import os
import pickle
from pathlib import Path

import numpy as np

import fiftyone as fo


DATASET_DIR = Path(
    "/mnt/NAS2/pjt_AI/00-git-archive/lidar-pytorch/data/seoulsi_yeoui"
)
PREDICTIONS_PATH = Path(
    "/mnt/NAS2/pjt_AI/00-git-archive/lidar-pytorch/output/kitti_models/"
    "IA-SSD_mix_seoulsi_yeoui_isaac260408_all_car_ground_aligned_intensity0_narrow_range/"
    "20260625-iassd-ga-intensity0-to-seoulsi-yeoui-intensity0-finetune/"
    "eval/epoch_0/val/eval_on_seoulsi_yeoui_intensity0/result.pkl"
)
COMPARE_PREDICTIONS_PATH = Path(
    "/mnt/NAS2/pjt_AI/00-git-archive/lidar-pytorch/output/kitti_models/"
    "IA-SSD_mix_seoulsi_yeoui_isaac260408_all_car_ground_aligned_narrow_range/"
    "20260623-iassd-isaac260408-all-car-ground-aligned-narrow-range/"
    "eval/epoch_260408/val/eval_on_seoulsi_yeoui/result.pkl"
)
TRAINING_DIR = DATASET_DIR / "training"
VELODYNE_DIR = TRAINING_DIR / "velodyne"
CALIB_DIR = TRAINING_DIR / "calib"
LABELS_DIR = TRAINING_DIR / "label_2"
IMAGES_DIR = TRAINING_DIR / "image_2"
IMAGESETS_DIR = DATASET_DIR / "ImageSets"
DATASET_NAME = "seoulsi_yeoui_3d_iassd"
PRED_FIELD = "predictions"
COMPARE_PRED_FIELD = "predictions_epoch260408"
EVAL_KEY = "eval_iassd"
COMPARE_EVAL_KEY = "eval_iassd_epoch260408"
MODEL_EVALUATION_PANEL = "model_evaluation_panel_builtin"
DEFAULT_DISTANCE_BINS = "0,20,40,60,80"
DISTANCE_SCENARIO_NAME = "Distance bins"


def _read_imageset(split):
    path = IMAGESETS_DIR / f"{split}.txt"
    if not path.exists():
        return []

    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def _available_ids(split):
    ids = _read_imageset(split)
    if not ids:
        ids = sorted(p.stem for p in VELODYNE_DIR.glob("*.bin"))

    return [
        kitti_id
        for kitti_id in ids
        if (VELODYNE_DIR / f"{kitti_id}.bin").exists()
    ]


def _load_calibration(calib_path):
    calib = {}
    with calib_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            key, values = line.split(":", 1)
            values = [float(v) for v in values.split()]

            if key.startswith("R") and len(values) == 9:
                calib[key] = np.asarray(values, dtype=np.float32).reshape(3, 3)
            elif key.startswith(("P", "T")) and len(values) == 12:
                calib[key] = np.asarray(values, dtype=np.float32).reshape(3, 4)

    return calib


def _lidar_to_camera_transform(calib):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :] = calib["R0_rect"] @ calib["Tr_velo_to_cam"]
    return transform


def _camera_point_to_lidar(point, camera_to_lidar):
    point_h = np.asarray([point[0], point[1], point[2], 1], dtype=np.float32)
    return (camera_to_lidar @ point_h)[:3]


def _kitti_rotation_to_lidar(rotation_y, camera_to_lidar):
    # KITTI boxes are yawed around camera Y. Convert their local length axis to
    # LiDAR coordinates, then keep only z-up yaw so boxes remain upright.
    c = np.cos(rotation_y)
    s = np.sin(rotation_y)
    length_axis_camera = np.asarray([c, 0, -s], dtype=np.float32)
    length_axis_lidar = camera_to_lidar[:3, :3] @ length_axis_camera
    return [0, 0, float(np.arctan2(length_axis_lidar[1], length_axis_lidar[0]))]


def _load_3d_detections(label_path, calib_path):
    if not label_path.exists():
        return None

    calib = _load_calibration(calib_path)
    camera_to_lidar = np.linalg.inv(_lidar_to_camera_transform(calib))

    detections = []
    with label_path.open() as f:
        for index, line in enumerate(f, 1):
            row = line.strip().split()
            if len(row) < 15 or row[0] == "DontCare":
                continue

            label = row[0]
            truncated = float(row[1])
            occluded = int(row[2])
            alpha = float(row[3])
            bbox_2d = [float(v) for v in row[4:8]]
            height, width, length = [float(v) for v in row[8:11]]
            x, y, z = [float(v) for v in row[11:14]]
            rotation_y = float(row[14])
            confidence = float(row[15]) if len(row) > 15 else None

            # KITTI stores object location as bottom-center in camera coords.
            center_camera = [x, y - 0.5 * height, z]
            center_lidar = _camera_point_to_lidar(center_camera, camera_to_lidar)

            detections.append(
                fo.Detection(
                    label=label,
                    location=[float(v) for v in center_lidar],
                    dimensions=[length, width, height],
                    rotation=_kitti_rotation_to_lidar(
                        rotation_y, camera_to_lidar
                    ),
                    confidence=confidence,
                    index=index,
                    truncated=truncated,
                    occluded=occluded,
                    alpha=alpha,
                    bbox_2d=bbox_2d,
                    rotation_y=rotation_y,
                )
            )

    return fo.Detections(detections=detections)


def _load_prediction_results(predictions_path):
    if not predictions_path.exists():
        return {}

    with predictions_path.open("rb") as f:
        records = pickle.load(f)

    predictions = {}
    for record in records:
        frame_id = str(record["frame_id"])
        boxes_lidar = record.get("boxes_lidar")
        names = record.get("name", [])
        scores = record.get("score", [])

        detections = []
        if boxes_lidar is not None:
            for index, box in enumerate(boxes_lidar, 1):
                x, y, z, dx, dy, dz, heading = [float(v) for v in box]
                label = str(names[index - 1]) if index - 1 < len(names) else None
                confidence = (
                    float(scores[index - 1]) if index - 1 < len(scores) else None
                )

                attrs = {}
                for key in ("alpha", "bbox", "dimensions", "location", "rotation_y"):
                    values = record.get(key)
                    if values is not None and index - 1 < len(values):
                        value = values[index - 1]
                        attrs[f"kitti_{key}"] = (
                            [float(v) for v in value]
                            if hasattr(value, "__iter__") and not isinstance(value, str)
                            else float(value)
                        )

                detections.append(
                    fo.Detection(
                        label=label,
                        location=[x, y, z],
                        dimensions=[dx, dy, dz],
                        rotation=[0, 0, heading],
                        confidence=confidence,
                        index=index,
                        **attrs,
                    )
                )

        predictions[frame_id] = fo.Detections(detections=detections)

    return predictions


def _load_velodyne_as_scene_points(kitti_id):
    velodyne_path = VELODYNE_DIR / f"{kitti_id}.bin"

    points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    intensity = points[:, 3:4]

    # Keep the LiDAR frame: x forward, y left, z up. This avoids baking camera
    # pitch/roll into the point cloud, which makes the ground look tilted.
    return np.concatenate([xyz, intensity], axis=1).astype(np.float32)


def _write_binary_pcd(pcd_path, points):
    pcd_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    ).encode("ascii")

    with pcd_path.open("wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(points, dtype=np.float32).tobytes())


def _write_fo3d(scene_path, pcd_path):
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    pcd_relpath = os.path.relpath(pcd_path, scene_path.parent)

    scene = fo.Scene(camera=fo.PerspectiveCamera(up="Z"))
    scene.add(
        fo.PointCloud(
            "velodyne",
            pcd_relpath,
            material=fo.PointCloudMaterial(
                shading_mode="intensity",
                point_size=2.0,
                attenuate_by_distance=False,
            ),
            flag_for_projection=True,
        )
    )
    scene.write(str(scene_path))


def _prepare_scene(kitti_id, pcd_dir, fo3d_dir, overwrite=False):
    pcd_path = pcd_dir / f"{kitti_id}.pcd"
    scene_path = fo3d_dir / f"{kitti_id}.fo3d"

    if overwrite or not pcd_path.exists():
        points = _load_velodyne_as_scene_points(kitti_id)
        _write_binary_pcd(pcd_path, points)

    if overwrite or not scene_path.exists():
        _write_fo3d(scene_path, pcd_path)

    return scene_path


def _tags_for_id(kitti_id, split_sets):
    return [split for split, ids in split_sets.items() if kitti_id in ids]


def _parse_distance_bins(value):
    edges = [float(v) for v in value.split(",") if v.strip()]
    if len(edges) < 2:
        raise ValueError("--distance-bins requires at least two comma-separated edges")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("--distance-bins edges must be strictly increasing")

    bins = []
    for lower, upper in zip(edges, edges[1:]):
        bins.append((lower, upper))

    bins.append((edges[-1], None))
    return bins


def _distance_bin_label(distance, bins):
    for lower, upper in bins:
        if upper is None:
            if distance >= lower:
                return f"{lower:g}m+"
        elif lower <= distance < upper:
            return f"{lower:g}-{upper:g}m"

    return f"<{bins[0][0]:g}m"


def _distance_bin_key(label):
    return (
        label.lower()
        .replace("+", "plus")
        .replace("-", "_")
        .replace(".", "p")
        .replace("m", "m")
    )


def _print_progress(prefix, index, total):
    if index == 1 or index == total or index % 100 == 0:
        print(f"{prefix}: {index}/{total}", end="\r", flush=True)
        if index == total:
            print()


def _prediction_sources():
    return {
        PRED_FIELD: PREDICTIONS_PATH,
        COMPARE_PRED_FIELD: COMPARE_PREDICTIONS_PATH,
    }


def _ensure_prediction_field(dataset, field_name, predictions_path, force=False):
    if dataset.has_sample_field(field_name) and not force:
        return

    prediction_map = _load_prediction_results(predictions_path)
    total = len(dataset)

    for index, sample in enumerate(dataset.iter_samples(autosave=True), 1):
        _print_progress(f"Loading {field_name}", index, total)
        sample[field_name] = prediction_map.get(
            sample["kitti_id"], fo.Detections(detections=[])
        )


def _ensure_prediction_fields(dataset, force=False):
    for field_name, predictions_path in _prediction_sources().items():
        _ensure_prediction_field(
            dataset, field_name, predictions_path, force=force
        )

    dataset.info["prediction_results_path"] = str(PREDICTIONS_PATH)
    dataset.info["compare_prediction_results_path"] = str(
        COMPARE_PREDICTIONS_PATH
    )
    dataset.save()


def build_dataset(ids, cache_dir, overwrite_assets=False):
    for path in (VELODYNE_DIR, CALIB_DIR, LABELS_DIR):
        if not path.is_dir():
            raise FileNotFoundError(f"Required dataset directory not found: {path}")

    pcd_dir = cache_dir / "pcd_lidar"
    fo3d_dir = cache_dir / "fo3d_lidar"
    prediction_maps = {
        field_name: _load_prediction_results(predictions_path)
        for field_name, predictions_path in _prediction_sources().items()
    }
    split_sets = {
        "all": set(_read_imageset("all")),
        "train": set(_read_imageset("train")),
        "val": set(_read_imageset("val")),
    }

    dataset = fo.Dataset(DATASET_NAME, persistent=True)
    dataset.app_config.plugins["3d"] = {
        "defaultCameraPosition": {"x": 0, "y": -80, "z": 50}
    }
    dataset.info["kitti_source_dir"] = str(DATASET_DIR)
    dataset.info["prediction_results_path"] = str(PREDICTIONS_PATH)
    dataset.info["compare_prediction_results_path"] = str(
        COMPARE_PREDICTIONS_PATH
    )
    dataset.info["pcd_cache_dir"] = str(pcd_dir)
    dataset.info["fo3d_cache_dir"] = str(fo3d_dir)
    dataset.save()

    samples = []
    total = len(ids)

    for index, kitti_id in enumerate(ids, 1):
        _print_progress("Preparing 3D scenes", index, total)
        scene_path = _prepare_scene(
            kitti_id, pcd_dir, fo3d_dir, overwrite=overwrite_assets
        )

        sample = fo.Sample(filepath=str(scene_path))
        sample["kitti_id"] = kitti_id
        sample["image_filepath"] = str(IMAGES_DIR / f"{kitti_id}.png")
        sample["velodyne_filepath"] = str(VELODYNE_DIR / f"{kitti_id}.bin")
        sample["calib_filepath"] = str(CALIB_DIR / f"{kitti_id}.txt")
        sample["ground_truth"] = _load_3d_detections(
            LABELS_DIR / f"{kitti_id}.txt", CALIB_DIR / f"{kitti_id}.txt"
        )
        for field_name, prediction_map in prediction_maps.items():
            sample[field_name] = prediction_map.get(
                kitti_id, fo.Detections(detections=[])
            )

        sample.tags = _tags_for_id(kitti_id, split_sets)
        samples.append(sample)

        if len(samples) >= 100:
            dataset.add_samples(samples)
            samples = []

    if samples:
        dataset.add_samples(samples)

    return dataset


def load_dataset(args):
    id_split = args.split if args.max_samples is not None else "all"
    ids = _available_ids(id_split)
    if args.max_samples is not None:
        ids = ids[: args.max_samples]

    cache_dir = Path(args.cache_dir) if args.cache_dir else TRAINING_DIR

    if fo.dataset_exists(DATASET_NAME) and not args.reload:
        dataset = fo.load_dataset(DATASET_NAME)
        _ensure_prediction_fields(dataset)
    else:
        if fo.dataset_exists(DATASET_NAME):
            fo.delete_dataset(DATASET_NAME)

        dataset = build_dataset(
            ids, cache_dir=cache_dir, overwrite_assets=args.overwrite_assets
        )

    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open seoulsi_yeoui point clouds, labels, and IA-SSD results."
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val"),
        default="val",
        help="Dataset split to show in the App",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Rebuild the FiftyOne dataset instead of reusing it",
    )
    parser.add_argument(
        "--overwrite-assets",
        action="store_true",
        help="Regenerate cached PCD/FO3D files",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory where pcd/ and fo3d/ caches are written. Defaults to "
            "the dataset training directory"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples when building the dataset",
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Build/load the dataset without launching the App",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run model evaluation",
    )
    parser.add_argument(
        "--no-eval-panel",
        action="store_true",
        help="Launch the App without the Model Evaluation panel",
    )
    parser.add_argument(
        "--eval-key",
        default=EVAL_KEY,
        help="Evaluation key to create/load",
    )
    parser.add_argument(
        "--compare-eval-key",
        default=COMPARE_EVAL_KEY,
        help="Evaluation key for the comparison predictions",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Do not create/select the comparison evaluation",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="3D IoU threshold for detection evaluation",
    )
    parser.add_argument(
        "--distance-bins",
        default=DEFAULT_DISTANCE_BINS,
        help=(
            "Comma-separated distance bin edges in meters. The final edge "
            "becomes an open-ended bin"
        ),
    )
    return parser.parse_args()


def _get_evaluation_id(dataset, eval_key):
    try:
        return str(dataset._doc.evaluations[eval_key].id)
    except Exception:
        return None


def _make_model_evaluation_panel(dataset, eval_key, compare_eval_key=None):
    evaluation_id = _get_evaluation_id(dataset, eval_key)
    compare_evaluation_id = (
        _get_evaluation_id(dataset, compare_eval_key)
        if compare_eval_key is not None
        else None
    )
    state = {}

    if evaluation_id is not None:
        view_state = {
            "page": "evaluation",
            "key": eval_key,
            "id": evaluation_id,
            "init": True,
        }
        if compare_evaluation_id is not None:
            view_state["compareKey"] = compare_eval_key

        state["view"] = view_state

    return fo.Panel(
        type=MODEL_EVALUATION_PANEL,
        pinned=True,
        state=state,
    )


def _make_app_spaces(
    dataset, eval_key, compare_eval_key=None, include_eval_panel=True
):
    samples_panel = fo.Panel(type="Samples", pinned=True)

    if not include_eval_panel:
        return fo.Space(children=[samples_panel])

    evaluation_panel = _make_model_evaluation_panel(
        dataset, eval_key, compare_eval_key=compare_eval_key
    )

    return fo.Space(
        children=[
            fo.Space(children=[samples_panel]),
            fo.Space(children=[evaluation_panel]),
        ],
        orientation="horizontal",
        sizes=[0.62, 0.38],
    )


def maybe_evaluate(dataset, pred_field, eval_key, args):
    if args.skip_eval:
        return None

    eval_view = dataset.match_tags("val")
    if dataset.has_evaluation(eval_key):
        info = dataset.get_evaluation_info(eval_key)
        if (
            info.config.pred_field == pred_field
            and info.config.gt_field == "ground_truth"
        ):
            return dataset.load_evaluation_results(eval_key)

        dataset.delete_evaluation(eval_key)

    results = eval_view.evaluate_detections(
        pred_field,
        gt_field="ground_truth",
        eval_key=eval_key,
        iou=args.iou,
        classwise=True,
        method="coco",
        compute_mAP=True,
    )
    return results


def _label_id(value):
    if value is None:
        return None

    value = str(value)
    if value in ("", "None", "nan"):
        return None

    return value


def _yaw(rotation):
    if rotation is None or len(rotation) < 3:
        return 0.0

    return float(rotation[2])


def _angle_error(yaw_a, yaw_b):
    delta = (yaw_a - yaw_b + np.pi) % (2 * np.pi) - np.pi
    return float(abs(delta))


def _scale_error(gt_detection, pred_detection):
    gt_dims = np.asarray(gt_detection.dimensions, dtype=np.float64)
    pred_dims = np.asarray(pred_detection.dimensions, dtype=np.float64)

    if np.any(gt_dims <= 0) or np.any(pred_dims <= 0):
        return None

    intersection = np.prod(np.minimum(gt_dims, pred_dims))
    union = np.prod(gt_dims) + np.prod(pred_dims) - intersection
    if union <= 0:
        return None

    return float(1.0 - intersection / union)


def _detection_distance(detection):
    location = detection.location
    if location is None or len(location) < 2:
        return None

    return float(np.linalg.norm(np.asarray(location[:2], dtype=np.float64)))


def _annotate_distance_bins(eval_view, bins, pred_field):
    gt_by_id = {}
    pred_by_id = {}

    for sample in eval_view:
        changed = False

        for field_name, index in (
            ("ground_truth", gt_by_id),
            (pred_field, pred_by_id),
        ):
            detections = sample[field_name]
            if detections is None:
                continue

            for detection in detections.detections:
                distance = _detection_distance(detection)
                if distance is not None:
                    detection.set_attribute_value("distance_m", distance)
                    detection.set_attribute_value(
                        "distance_bin", _distance_bin_label(distance, bins)
                    )
                    changed = True

                index[str(detection.id)] = detection

        if changed:
            sample.save()

    return gt_by_id, pred_by_id


def _ensure_distance_fields(dataset, pred_fields):
    fields = [
        ("ground_truth.detections.distance_m", fo.FloatField),
        ("ground_truth.detections.distance_bin", fo.StringField),
    ]

    for pred_field in pred_fields:
        fields.extend(
            [
                (f"{pred_field}.detections.distance_m", fo.FloatField),
                (f"{pred_field}.detections.distance_bin", fo.StringField),
            ]
        )

    for path, field_type in fields:
        if not dataset.has_sample_field(path):
            dataset.add_sample_field(path, field_type)


def _delete_obsolete_iou_fields(dataset, eval_key, pred_field):
    label_attr = f"{eval_key}_max_iou"
    fields = [
        f"ground_truth.detections.{label_attr}",
        f"{pred_field}.detections.{label_attr}",
        f"ground_truth.detections.{eval_key}_iou",
        f"{pred_field}.detections.{eval_key}_iou",
    ]

    for prefix in (f"{eval_key}_pred", f"{eval_key}_gt"):
        fields.extend(
            [
                f"{prefix}_min_iou",
                f"{prefix}_mean_iou",
                f"{prefix}_max_iou",
                f"{prefix}_low_iou_count",
                f"{prefix}_has_low_iou",
            ]
        )

    existing = [path for path in fields if dataset.has_sample_field(path)]
    if existing:
        dataset.delete_sample_fields(existing, error_level=2)

    dataset.info.pop(f"{eval_key}_iou_filter_pred_field", None)
    dataset.info.pop(f"{eval_key}_low_iou_threshold", None)


def _ensure_prediction_iou_field(dataset, pred_field):
    path = f"{pred_field}.detections.iou"
    if not dataset.has_sample_field(path):
        dataset.add_sample_field(path, fo.FloatField)


def _prediction_max_ious(predictions, ground_truth):
    if not predictions:
        return []

    if not ground_truth:
        return [0.0] * len(predictions)

    import fiftyone.utils.iou as foui

    ious = foui.compute_ious(
        predictions,
        ground_truth,
        classwise=True,
        error_level=1,
    )
    if ious.shape[1] == 0:
        return [0.0] * len(predictions)

    return [float(value) for value in np.max(ious, axis=1)]


def maybe_add_iou_filter_field(dataset, results, pred_field):
    if results is None:
        return None

    eval_key = results.key
    _delete_obsolete_iou_fields(dataset, eval_key, pred_field)
    _ensure_prediction_iou_field(dataset, pred_field)

    eval_view = dataset.match_tags("val")
    total = len(eval_view)
    for index, sample in enumerate(eval_view.iter_samples(autosave=True), 1):
        _print_progress(f"Computing {pred_field} IoU", index, total)
        prediction_doc = sample[pred_field]
        ground_truth_doc = sample["ground_truth"]
        predictions = [] if prediction_doc is None else prediction_doc.detections
        ground_truth = [] if ground_truth_doc is None else ground_truth_doc.detections
        ious = _prediction_max_ious(predictions, ground_truth)

        for detection, iou in zip(predictions, ious):
            detection.set_attribute_value("iou", iou)

    dataset.info[f"{pred_field}_iou_field"] = f"{pred_field}.detections.iou"
    dataset.save()
    return "iou"


def _safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None

    return float(np.mean(values))


def _safe_ratio(numerator, denominator):
    if denominator <= 0:
        return None

    return float(numerator / denominator)


def _fscore(precision, recall):
    if precision is None or recall is None or precision + recall <= 0:
        return None

    return float(2 * precision * recall / (precision + recall))


def _custom_metric(uri, key, label, value, lower_is_better=True):
    return {
        uri: {
            "key": key,
            "value": value,
            "label": label,
            "lower_is_better": lower_is_better,
        }
    }


def _compute_additional_metrics(eval_view, results, bins, pred_field):
    gt_by_id, pred_by_id = _annotate_distance_bins(eval_view, bins, pred_field)

    bin_labels = [_distance_bin_label(lower, bins) for lower, _ in bins]
    stats = {
        label: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ase": [],
            "aoe": [],
        }
        for label in bin_labels
    }

    all_ase = []
    all_aoe = []

    for gt_id, pred_id in zip(results.ytrue_ids, results.ypred_ids):
        gt_id = _label_id(gt_id)
        pred_id = _label_id(pred_id)

        gt_detection = gt_by_id.get(gt_id) if gt_id is not None else None
        pred_detection = pred_by_id.get(pred_id) if pred_id is not None else None

        if gt_detection is not None:
            distance = _detection_distance(gt_detection)
        elif pred_detection is not None:
            distance = _detection_distance(pred_detection)
        else:
            distance = None

        if distance is None:
            continue

        bin_label = _distance_bin_label(distance, bins)
        bin_stats = stats.setdefault(
            bin_label, {"tp": 0, "fp": 0, "fn": 0, "ase": [], "aoe": []}
        )

        if gt_detection is not None and pred_detection is not None:
            bin_stats["tp"] += 1

            ase = _scale_error(gt_detection, pred_detection)
            aoe = _angle_error(_yaw(gt_detection.rotation), _yaw(pred_detection.rotation))
            bin_stats["ase"].append(ase)
            bin_stats["aoe"].append(aoe)
            all_ase.append(ase)
            all_aoe.append(aoe)
        elif gt_detection is None and pred_detection is not None:
            bin_stats["fp"] += 1
        elif gt_detection is not None and pred_detection is None:
            bin_stats["fn"] += 1

    metrics = {}
    metrics.update(
        _custom_metric(
            "script/iassd_ase",
            "ase",
            "ASE",
            _safe_mean(all_ase),
            lower_is_better=True,
        )
    )
    metrics.update(
        _custom_metric(
            "script/iassd_aoe",
            "aoe",
            "AOE (rad)",
            _safe_mean(all_aoe),
            lower_is_better=True,
        )
    )

    for bin_label, bin_stats in stats.items():
        tp = bin_stats["tp"]
        fp = bin_stats["fp"]
        fn = bin_stats["fn"]
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        fscore = _fscore(precision, recall)
        key_prefix = f"distance_{_distance_bin_key(bin_label)}"
        label_prefix = f"{bin_label}"

        metrics.update(
            _custom_metric(
                f"script/{key_prefix}_precision",
                f"{key_prefix}_precision",
                f"Precision @ {label_prefix}",
                precision,
                lower_is_better=False,
            )
        )
        metrics.update(
            _custom_metric(
                f"script/{key_prefix}_recall",
                f"{key_prefix}_recall",
                f"Recall @ {label_prefix}",
                recall,
                lower_is_better=False,
            )
        )
        metrics.update(
            _custom_metric(
                f"script/{key_prefix}_fscore",
                f"{key_prefix}_fscore",
                f"F1 @ {label_prefix}",
                fscore,
                lower_is_better=False,
            )
        )
        metrics.update(
            _custom_metric(
                f"script/{key_prefix}_ase",
                f"{key_prefix}_ase",
                f"ASE @ {label_prefix}",
                _safe_mean(bin_stats["ase"]),
                lower_is_better=True,
            )
        )
        metrics.update(
            _custom_metric(
                f"script/{key_prefix}_aoe",
                f"{key_prefix}_aoe",
                f"AOE @ {label_prefix}",
                _safe_mean(bin_stats["aoe"]),
                lower_is_better=True,
            )
        )

    results.custom_metrics = metrics
    results.save()
    return metrics


def _ensure_distance_scenario(dataset, eval_key, bins):
    from bson import ObjectId
    from fiftyone.operators.store import ExecutionStore

    eval_view = dataset.load_evaluation_view(eval_key)
    counts = eval_view.count_values("ground_truth.detections.distance_bin")
    counts.pop(None, None)

    ordered_subsets = [
        _distance_bin_label(lower, bins)
        for lower, _ in bins
        if _distance_bin_label(lower, bins) in counts
    ]
    if not ordered_subsets:
        return None

    store = ExecutionStore.create(MODEL_EVALUATION_PANEL, dataset._doc.id)
    scenarios = (store.get("scenarios") or {}).copy()
    scenario_id = None

    for existing_id, scenario in scenarios.items():
        if scenario.get("name") == DISTANCE_SCENARIO_NAME:
            scenario_id = existing_id
            break

    if scenario_id is None:
        scenario_id = str(ObjectId())

    scenarios[scenario_id] = {
        "id": scenario_id,
        "name": DISTANCE_SCENARIO_NAME,
        "type": "label_attribute",
        "field": "ground_truth.detections.distance_bin",
        "subsets": ordered_subsets,
    }
    store.set("scenarios", scenarios)
    store.clear_cache()
    return scenario_id


def maybe_add_additional_metrics(dataset, results, pred_field, args):
    if results is None:
        return None

    bins = _parse_distance_bins(args.distance_bins)
    _ensure_distance_fields(dataset, [PRED_FIELD, COMPARE_PRED_FIELD])
    eval_view = dataset.match_tags("val")
    metrics = _compute_additional_metrics(eval_view, results, bins, pred_field)
    _ensure_distance_scenario(dataset, results.key, bins)
    return metrics


def main():
    args = parse_args()
    dataset = load_dataset(args)
    results = maybe_evaluate(dataset, PRED_FIELD, args.eval_key, args)
    compare_results = None
    if not args.no_compare:
        compare_results = maybe_evaluate(
            dataset, COMPARE_PRED_FIELD, args.compare_eval_key, args
        )

    extra_metrics = maybe_add_additional_metrics(
        dataset, results, PRED_FIELD, args
    )
    iou_filter_attr = maybe_add_iou_filter_field(
        dataset, results, PRED_FIELD
    )
    compare_extra_metrics = None
    compare_iou_filter_attr = None
    if compare_results is not None:
        compare_extra_metrics = maybe_add_additional_metrics(
            dataset, compare_results, COMPARE_PRED_FIELD, args
        )
        compare_iou_filter_attr = maybe_add_iou_filter_field(
            dataset, compare_results, COMPARE_PRED_FIELD
        )

    view = dataset if args.split == "all" else dataset.match_tags(args.split)

    print(dataset)
    print(f"Showing split: {args.split} ({len(view)} samples)")
    print("Label field: ground_truth")
    print(f"Prediction field: {PRED_FIELD}")
    if not args.no_compare:
        print(f"Compare prediction field: {COMPARE_PRED_FIELD}")
    if results is not None:
        print(f"Evaluation key: {args.eval_key}")
    if compare_results is not None:
        print(f"Compare evaluation key: {args.compare_eval_key}")
    if extra_metrics is not None:
        print(f"Additional metrics: {len(extra_metrics)}")
    if compare_extra_metrics is not None:
        print(f"Compare additional metrics: {len(compare_extra_metrics)}")
    if iou_filter_attr is not None:
        print(f"IoU filter: {PRED_FIELD}.detections.{iou_filter_attr}")
    if compare_iou_filter_attr is not None:
        print(
            "Compare IoU filter: "
            f"{COMPARE_PRED_FIELD}.detections.{compare_iou_filter_attr}"
        )
    if not args.no_eval_panel:
        evaluation_id = _get_evaluation_id(dataset, args.eval_key)
        if evaluation_id is not None:
            print(f"Model Evaluation panel: {args.eval_key}")
        else:
            print("Model Evaluation panel: opened without selected evaluation")

    if args.no_app:
        return

    spaces = _make_app_spaces(
        dataset,
        args.eval_key,
        compare_eval_key=None if args.no_compare else args.compare_eval_key,
        include_eval_panel=not args.no_eval_panel,
    )
    session = fo.launch_app(view, spaces=spaces)
    session.wait()


if __name__ == "__main__":
    main()
