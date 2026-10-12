"""Video segmentation, frame extraction, and reference-image cropping."""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any

import cv2
import shutil
import subprocess
from pathlib import Path


def duration_seconds(path: Path) -> float:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
        capture_output=True, text=True, check=True,
    )
    value = float(result.stdout.strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid video duration: {path}')
    return value


def split_video_with_retry(
    video_path, save_base_dir, target_seconds=120, max_len=125, max_retry=10,
):
    """Use the research splitter's keyframe boundaries and stream-copy retries.

    The returned segment length/count are nominal, as in the research splitter;
    FFmpeg's actual clip count can differ because cuts occur at keyframes.
    """
    for binary in ('ffmpeg', 'ffprobe'):
        if not shutil.which(binary):
            raise RuntimeError(f'{binary} is required; install FFmpeg first')
    src = Path(video_path).resolve()
    if target_seconds <= 0 or max_len < target_seconds or max_retry <= 0:
        raise ValueError('Expected 0 < target_seconds <= max_len and max_retry > 0')
    duration = duration_seconds(src)
    out = Path(save_base_dir) / src.stem
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'cache.json'
    stat = src.stat()
    key = {'source': str(src), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
           'target_seconds': target_seconds, 'max_len': max_len,
           'max_retry': max_retry}

    def valid_clips(clips):
        durations = [duration_seconds(p) for p in clips]
        return (bool(durations) and all(d <= max_len for d in durations)
                and abs(sum(durations) - duration) <= 5)

    try:
        cache = json.loads(manifest.read_text())
        clips = sorted(out.glob('clip_*.mp4'))
        if (cache['key'] == key
                and cache['sizes'] == {p.name: p.stat().st_size for p in clips}
                and valid_clips(clips)):
            return cache['segment_len'], cache['n_segments'], str(out)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        pass

    # Remove only splitter-owned artifacts before regenerating clips.
    manifest.unlink(missing_ok=True)
    for _ in range(max_retry):
        for old in out.glob('clip_*.mp4'):
            old.unlink()
        if target_seconds <= 0:
            break
        n_segments = max(1, math.ceil(duration / target_seconds))
        segment_len = max(1, int(math.ceil(duration / n_segments)))
        min_seg_duration = max(1, int(segment_len * 0.7))
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', str(src), '-c', 'copy', '-map', '0',
             '-f', 'segment', '-segment_time', str(segment_len),
             '-min_seg_duration', str(min_seg_duration),
             '-reset_timestamps', '1', '-segment_start_number', '1',
             str(out / 'clip_%02d.mp4')],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f'FFmpeg segmentation failed: {result.stderr}')
        clips = sorted(out.glob('clip_*.mp4'))
        if valid_clips(clips):
            manifest.write_text(json.dumps({
                'key': key, 'segment_len': segment_len, 'n_segments': n_segments,
                'sizes': {p.name: p.stat().st_size for p in clips},
            }))
            return segment_len, n_segments, str(out)
        target_seconds = int(target_seconds * 0.95)
    for old in out.glob('clip_*.mp4'):
        old.unlink()
    raise RuntimeError('Failed to split video within duration constraints')


FRAMES_DIR_NAME = "04_frames"
CROPS_DIR_NAME = "05_crops"


def clamp_bbox_1000(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_1000 must contain four values")
    try:
        ymin, xmin, ymax, xmax = [
            min(1000, max(0, int(round(float(item))))) for item in value
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bbox_1000: {value!r}") from exc
    if ymax <= ymin or xmax <= xmin:
        raise ValueError(f"Empty bbox_1000: {[ymin, xmin, ymax, xmax]}")
    return [ymin, xmin, ymax, xmax]


def bbox_to_pixels(bbox: list[int], width: int, height: int) -> list[int]:
    ymin, xmin, ymax, xmax = clamp_bbox_1000(bbox)
    x1 = max(0, min(width - 1, math.floor(xmin * width / 1000)))
    y1 = max(0, min(height - 1, math.floor(ymin * height / 1000)))
    x2 = max(x1 + 1, min(width, math.ceil(xmax * width / 1000)))
    y2 = max(y1 + 1, min(height, math.ceil(ymax * height / 1000)))
    return [x1, y1, x2, y2]


def safe_subject_name(subject_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", subject_id).strip("_") or "subject"


def probe_video(video_path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid FPS/frame count for {video_path}")
    return fps, frame_count / fps, frame_count


def _read_frame(video_path: Path, timestamp: Any) -> tuple[float, Any]:
    fps, duration, frame_count = probe_video(video_path)
    timestamp = float(timestamp)
    if not math.isfinite(timestamp):
        raise ValueError("timestamp_sec must be finite")
    timestamp = min(max(timestamp, 0.0), max(duration - 1.0 / fps, 0.0))
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(round(timestamp * fps), frame_count - 1))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read {video_path} at {timestamp:.3f}s")
    finally:
        capture.release()
    return timestamp, frame


def extract_frame(
    video_path: Path,
    detection: dict[str, Any],
    detection_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    timestamp, frame = _read_frame(video_path, detection["timestamp_sec"])

    name = safe_subject_name(str(detection["subject_id"]))
    stem = f"{detection_index:03d}_{name}_{timestamp:.3f}"
    frame_path = output_dir / FRAMES_DIR_NAME / f"{stem}.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"Cannot write frame: {frame_path}")
    result = deepcopy(detection)
    result.update(
        {
            "detection_index": detection_index,
            "timestamp_sec": timestamp,
            "frame_path": str(frame_path.resolve()),
        }
    )
    return result


def extract_and_crop(
    video_path: Path,
    detection: dict[str, Any],
    detection_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    timestamp, frame = _read_frame(video_path, detection["timestamp_sec"])

    height, width = frame.shape[:2]
    bbox_1000 = clamp_bbox_1000(detection["bbox_1000"])
    bbox_pixels = bbox_to_pixels(bbox_1000, width, height)
    x1, y1, x2, y2 = bbox_pixels
    if (x2 - x1) * (y2 - y1) < 16:
        raise ValueError(f"Crop is too small: {bbox_pixels}")
    name = safe_subject_name(str(detection["subject_id"]))
    stem = f"{detection_index:03d}_{name}_{timestamp:.3f}"
    frame_path = output_dir / FRAMES_DIR_NAME / f"{stem}.jpg"
    crop_path = output_dir / CROPS_DIR_NAME / f"{stem}.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"Cannot write frame: {frame_path}")
    if not cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2]):
        raise RuntimeError(f"Cannot write crop: {crop_path}")
    result = deepcopy(detection)
    result.update(
        {
            "detection_index": detection_index,
            "timestamp_sec": timestamp,
            "bbox_1000": bbox_1000,
            "bbox_pixels": bbox_pixels,
            "frame_size": [width, height],
            "frame_path": str(frame_path.resolve()),
            "crop_path": str(crop_path.resolve()),
        }
    )
    return result

