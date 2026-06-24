from __future__ import annotations


import argparse
import json
import math


from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Sample:
    sample_idx: int
    frame_idx: int
    time_sec: float
    hist: np.ndarray
    phash: int
    blur: float
    entropy: float
    brightness: float
    diff_prev: float
    cut_strength: float = 0.0
    scene_id: int = -1
    chunk_id: int = -1
    quality_score: float = 0.0
    importance_score: float = 0.0
    select_score: float = 0.0


@dataclass
class Candidate:
    sample_idx: int
    frame_idx: int
    time_sec: float
    scene_id: int
    chunk_id: int
    quality_score: float
    importance_score: float
    select_score: float
    phash: int
    hist: np.ndarray


@dataclass
class Scene:
    scene_id: int
    start_idx: int
    end_idx: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    variability: float
    cut_before: float


@dataclass
class Chunk:
    scene_id: int
    chunk_id: int
    start_idx: int
    end_idx: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float


EPS = 1e-8


def positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return x


def nonnegative_int(value: str) -> int:
    x = int(value)
    if x < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return x


def robust_normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = np.percentile(arr, 5)
    hi = np.percentile(arr, 95)
    if hi - lo < EPS:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def resize_keep_aspect(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame.copy()
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + EPS)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist + EPS)).sum())


def compute_phash(gray: np.ndarray) -> int:
    # Standard pHash: 32x32 -> DCT -> top-left 8x8 (excluding DC via median thresholding).
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    block = dct[:8, :8].flatten()
    med = float(np.median(block[1:]))
    bits = 0
    for i, value in enumerate(block):
        if value > med:
            bits |= 1 << i
    return bits


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def histogram_for_frame(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 12], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    score = float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(1.0, score))


def sample_video(video_path: Path, sample_fps: float, thumb_max_side: int) -> tuple[list[Sample], float, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        # Fallback to manual counting if metadata is missing.
        fps = fps if fps > 0 else 30.0

    duration_sec = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    stride = max(1, int(round(fps / sample_fps)))

    samples: list[Sample] = []
    prev_gray = None
    prev_edges = None
    prev_hist = None
    frame_idx = -1
    sample_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        thumb = resize_keep_aspect(frame, thumb_max_side)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        hist = histogram_for_frame(thumb)
        blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        entropy = compute_entropy(gray)
        brightness = float(gray.mean() / 255.0)
        phash = compute_phash(gray)

        diff_prev = 0.0
        if prev_gray is not None and prev_edges is not None and prev_hist is not None:
            gray_diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            edge_diff = float(np.mean(cv2.absdiff(edges, prev_edges)) / 255.0)
            hist_diff = histogram_distance(hist, prev_hist)
            diff_prev = 0.50 * hist_diff + 0.35 * gray_diff + 0.15 * edge_diff

        samples.append(
            Sample(
                sample_idx=sample_idx,
                frame_idx=frame_idx,
                time_sec=frame_idx / fps,
                hist=hist,
                phash=phash,
                blur=blur,
                entropy=entropy,
                brightness=brightness,
                diff_prev=diff_prev,
            )
        )
        prev_gray = gray
        prev_edges = edges
        prev_hist = hist
        sample_idx += 1

    cap.release()
    return samples, fps, duration_sec, frame_count, stride


def detect_scene_cuts(samples: list[Sample], sample_fps: float, min_scene_sec: float) -> list[int]:
    if len(samples) < 3:
        return []

    scores = np.array([s.diff_prev for s in samples], dtype=np.float32)
    global_median = float(np.median(scores[1:])) if len(scores) > 1 else 0.0
    global_mad = float(np.median(np.abs(scores[1:] - global_median))) + EPS if len(scores) > 1 else 0.0
    # global_abs_thresh = max(0.16, global_median + 3.0 * global_mad)
    global_abs_thresh = max(0.08, global_median + 2.0 * global_mad)
    min_gap_samples = max(1, int(round(min_scene_sec * sample_fps)))

    cuts: list[int] = []
    last_cut = 0
    for i in range(1, len(samples) - 1):
        score = float(scores[i])
        if i - last_cut < min_gap_samples:
            continue
        if score < global_abs_thresh:
            continue
        if not (score >= scores[i - 1] and score >= scores[i + 1]):
            continue

        lo = max(1, i - 6)
        hi = min(len(scores), i + 7)
        neighbors = np.concatenate([scores[lo:i], scores[i + 1:hi]])
        if neighbors.size == 0:
            neighbors = scores[1:]
        local_mean = float(np.mean(neighbors)) + EPS
        local_median = float(np.median(neighbors))
        ratio = score / local_mean
        if ratio < 1.75 and score < local_median + 0.10:
            continue

        samples[i].cut_strength = score
        cuts.append(i)
        last_cut = i

    return cuts


def build_scenes(
    samples: list[Sample],
    cut_indices: list[int],
    dynamic_max_gap_sec: float,
    static_max_gap_sec: float,
) -> tuple[list[Scene], list[Chunk]]:
    if not samples:
        return [], []

    boundaries = [0] + cut_indices + [len(samples) - 1]
    scenes: list[Scene] = []
    raw_variabilities: list[float] = []

    for scene_id in range(len(boundaries) - 1):
        start_idx = boundaries[scene_id]
        end_idx = boundaries[scene_id + 1]
        if end_idx <= start_idx:
            continue
        start_time = samples[start_idx].time_sec
        end_time = samples[end_idx].time_sec
        duration = max(0.001, end_time - start_time)
        diffs = [samples[i].diff_prev for i in range(start_idx + 1, end_idx + 1)]
        variability = float(np.mean(diffs)) if diffs else 0.0
        raw_variabilities.append(variability)
        cut_before = float(samples[start_idx].cut_strength) if start_idx < len(samples) else 0.0
        scenes.append(
            Scene(
                scene_id=scene_id,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time_sec=start_time,
                end_time_sec=end_time,
                duration_sec=duration,
                variability=variability,
                cut_before=cut_before,
            )
        )

    if not scenes:
        whole = Scene(0, 0, len(samples) - 1, samples[0].time_sec, samples[-1].time_sec, max(0.001, samples[-1].time_sec - samples[0].time_sec), 0.0, 0.0)
        scenes = [whole]
        raw_variabilities = [0.0]

    variability_norm = robust_normalize(raw_variabilities)
    chunks: list[Chunk] = []
    chunk_id = 0
    for scene, var_norm in zip(scenes, variability_norm):
        effective_gap = float(static_max_gap_sec - (static_max_gap_sec - dynamic_max_gap_sec) * var_norm)
        effective_gap = max(dynamic_max_gap_sec, min(static_max_gap_sec, effective_gap))
        splits = max(1, int(math.ceil(scene.duration_sec / max(effective_gap, EPS))))
        if splits == 1:
            chunk = Chunk(
                scene_id=scene.scene_id,
                chunk_id=chunk_id,
                start_idx=scene.start_idx,
                end_idx=scene.end_idx,
                start_time_sec=scene.start_time_sec,
                end_time_sec=scene.end_time_sec,
                duration_sec=scene.duration_sec,
            )
            chunks.append(chunk)
            chunk_id += 1
            continue

        total_samples = scene.end_idx - scene.start_idx + 1
        for split_idx in range(splits):
            sub_start = scene.start_idx + int(round(split_idx * total_samples / splits))
            sub_end = scene.start_idx + int(round((split_idx + 1) * total_samples / splits)) - 1
            sub_end = min(sub_end, scene.end_idx)
            sub_start = min(sub_start, sub_end)
            start_time = samples[sub_start].time_sec
            end_time = samples[sub_end].time_sec
            chunk = Chunk(
                scene_id=scene.scene_id,
                chunk_id=chunk_id,
                start_idx=sub_start,
                end_idx=sub_end,
                start_time_sec=start_time,
                end_time_sec=end_time,
                duration_sec=max(0.001, end_time - start_time),
            )
            chunks.append(chunk)
            chunk_id += 1

    return scenes, chunks


def choose_representative_frames(samples: list[Sample], scenes: list[Scene], chunks: list[Chunk]) -> list[Candidate]:
    if not samples or not chunks:
        return []

    blur_norm = robust_normalize(np.log1p([s.blur for s in samples]))
    entropy_norm = robust_normalize([s.entropy for s in samples])
    motion_norm = robust_normalize([(s.diff_prev + (samples[s.sample_idx + 1].diff_prev if s.sample_idx + 1 < len(samples) else s.diff_prev)) / 2.0 for s in samples])

    candidates: list[Candidate] = []
    max_scene_duration = max((scene.duration_sec for scene in scenes), default=1.0)
    max_scene_var = max((scene.variability for scene in scenes), default=1.0) + EPS

    scene_map = {scene.scene_id: scene for scene in scenes}

    for chunk in chunks:
        idxs = list(range(chunk.start_idx, chunk.end_idx + 1))
        if not idxs:
            continue

        # Avoid transition frames near hard cuts when enough alternatives exist.
        interior = idxs[:]
        if len(interior) >= 3:
            interior = interior[1:-1]
        if not interior:
            interior = idxs

        chunk_hists = np.stack([samples[i].hist for i in interior], axis=0)
        mean_hist = np.mean(chunk_hists, axis=0).astype(np.float32)

        best_idx = interior[0]
        best_score = -1.0
        for rank_pos, i in enumerate(interior):
            sample = samples[i]
            sample.scene_id = chunk.scene_id
            sample.chunk_id = chunk.chunk_id

            quality = 0.70 * float(blur_norm[i]) + 0.30 * float(entropy_norm[i])
            stability = 1.0 - float(motion_norm[i])
            brightness = sample.brightness
            exposure = max(0.0, 1.0 - min(1.0, abs(brightness - 0.50) / 0.50))
            represent = 1.0 - histogram_distance(sample.hist, mean_hist)
            if len(interior) == 1:
                centrality = 1.0
            else:
                center = (len(interior) - 1) / 2.0
                centrality = 1.0 - abs(rank_pos - center) / max(center, 1.0)

            scene = scene_map.get(chunk.scene_id)
            duration_term = (scene.duration_sec / max_scene_duration) if scene else 0.0
            variability_term = (scene.variability / max_scene_var) if scene else 0.0
            cut_term = min(1.0, (scene.cut_before / 0.40)) if scene else 0.0
            importance = 0.45 * duration_term + 0.35 * variability_term + 0.20 * cut_term
            select_score = 0.32 * quality + 0.24 * stability + 0.20 * represent + 0.14 * exposure + 0.10 * centrality
            select_score += 0.10 * importance

            sample.quality_score = quality
            sample.importance_score = importance
            sample.select_score = select_score

            if select_score > best_score:
                best_score = select_score
                best_idx = i

        best = samples[best_idx]
        candidates.append(
            Candidate(
                sample_idx=best.sample_idx,
                frame_idx=best.frame_idx,
                time_sec=best.time_sec,
                scene_id=best.scene_id,
                chunk_id=best.chunk_id,
                quality_score=best.quality_score,
                importance_score=best.importance_score,
                select_score=best.select_score,
                phash=best.phash,
                hist=best.hist,
            )
        )

    candidates.sort(key=lambda c: c.time_sec)
    return candidates


def auto_budget(
    duration_sec: float,
    target_seconds_per_frame: float,
    max_auto_frames: int,
    scene_count: int,
    candidate_count: int,
) -> int:
    if duration_sec <= 0:
        return max(1, min(max_auto_frames, candidate_count if candidate_count > 0 else 1))
    estimate = int(math.ceil(duration_sec / max(target_seconds_per_frame, 0.1)))
    scene_floor = int(math.ceil(scene_count * 0.75))
    candidate_floor = int(math.ceil(candidate_count * 0.60))
    return max(4, min(max_auto_frames, max(estimate, scene_floor, candidate_floor)))


def compress_to_budget(candidates: list[Candidate], duration_sec: float, budget: int) -> list[Candidate]:
    if budget <= 0 or len(candidates) <= budget:
        return candidates[:]
    bin_size = max(duration_sec / budget, 0.001)
    chosen_by_bin: dict[int, Candidate] = {}
    for cand in candidates:
        bin_idx = min(budget - 1, int(cand.time_sec / bin_size))
        current = chosen_by_bin.get(bin_idx)
        cand_strength = 0.60 * cand.importance_score + 0.40 * cand.select_score
        if current is None:
            chosen_by_bin[bin_idx] = cand
        else:
            cur_strength = 0.60 * current.importance_score + 0.40 * current.select_score
            if cand_strength > cur_strength:
                chosen_by_bin[bin_idx] = cand

    selected = sorted(chosen_by_bin.values(), key=lambda c: c.time_sec)
    if len(selected) >= budget:
        return selected[:budget]

    used = {(c.scene_id, c.chunk_id, c.frame_idx) for c in selected}
    remaining = [c for c in candidates if (c.scene_id, c.chunk_id, c.frame_idx) not in used]
    remaining.sort(key=lambda c: (0.60 * c.importance_score + 0.40 * c.select_score), reverse=True)
    for cand in remaining:
        if len(selected) >= budget:
            break
        selected.append(cand)
    selected.sort(key=lambda c: c.time_sec)
    return selected[:budget]


def deduplicate(candidates: list[Candidate], min_time_gap_sec: float, phash_threshold: int, hist_threshold: float) -> list[Candidate]:
    kept: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: c.time_sec):
        duplicate = False
        for prev in kept:
            time_gap = abs(cand.time_sec - prev.time_sec)
            if time_gap > min_time_gap_sec:
                continue
            hash_gap = hamming_distance(cand.phash, prev.phash)
            hist_gap = histogram_distance(cand.hist, prev.hist)
            if hash_gap <= phash_threshold or hist_gap <= hist_threshold:
                prev_strength = 0.55 * prev.select_score + 0.45 * prev.importance_score
                cand_strength = 0.55 * cand.select_score + 0.45 * cand.importance_score
                if cand_strength > prev_strength:
                    kept.remove(prev)
                    kept.append(cand)
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    kept.sort(key=lambda c: c.time_sec)
    return kept

def grab_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame


def save_optimized_jpeg(frame_bgr: np.ndarray, out_path: Path, max_dimension: int, jpeg_quality: int) -> tuple[int, int]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    if max_dimension > 0 and max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    image.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
    return image.size


def build_contact_sheet(output_dir: Path, image_paths: list[Path], thumb_size: int = 320, columns: int = 4) -> Path | None:
    if not image_paths:
        return None
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    label_h = 20
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_size, thumb_size + label_h), "white")
        x = (thumb_size - thumb.width) // 2
        y = (thumb_size - thumb.height) // 2
        tile.paste(thumb, (x, y))
        draw = ImageDraw.Draw(tile)
        draw.text((5, thumb_size + 3), path.stem[:40], fill="black", font=font)
        tiles.append(tile)

    rows = int(math.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_h)), "#f5f5f5")
    for i, tile in enumerate(tiles):
        x = (i % columns) * thumb_size
        y = (i // columns) * (thumb_size + label_h)
        sheet.paste(tile, (x, y))

    out_path = output_dir / "contact_sheet.jpg"
    sheet.save(out_path, format="JPEG", quality=90, optimize=True, progressive=True)
    return out_path


def extract_keyframes(args: argparse.Namespace) -> Path:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # default_out = video_path.parent / f"{video_path.stem}_keyframes"
    default_out = Path("frames") / video_path.stem
    output_dir = Path(args.output).expanduser().resolve() if args.output else default_out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, fps, duration_sec, frame_count, stride = sample_video(video_path, args.sample_fps, args.thumb_max_side)
    if not samples:
        raise RuntimeError("No frames could be sampled from the video.")

    cut_indices = detect_scene_cuts(samples, args.sample_fps, args.min_scene_sec)
    scenes, chunks = build_scenes(samples, cut_indices, args.dynamic_max_gap_sec, args.static_max_gap_sec)
    candidates = choose_representative_frames(samples, scenes, chunks)

    budget = args.max_frames if args.max_frames > 0 else auto_budget(duration_sec, args.target_seconds_per_frame, args.max_auto_frames, len(scenes), len(candidates))
    candidates = compress_to_budget(candidates, duration_sec, budget)
    candidates = deduplicate(candidates, args.min_output_gap_sec, args.phash_threshold, args.hist_threshold)

    # If dedup becomes too aggressive for extremely short videos, keep at least one frame.
    if not candidates:
        mid = samples[len(samples) // 2]
        candidates = [
            Candidate(
                sample_idx=mid.sample_idx,
                frame_idx=mid.frame_idx,
                time_sec=mid.time_sec,
                scene_id=mid.scene_id,
                chunk_id=mid.chunk_id,
                quality_score=mid.quality_score,
                importance_score=mid.importance_score,
                select_score=mid.select_score,
                phash=mid.phash,
                hist=mid.hist,
            )
        ]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video for extraction: {video_path}")

    saved_paths: list[Path] = []
    metadata_frames: list[dict] = []

    for rank, cand in enumerate(sorted(candidates, key=lambda c: c.time_sec), start=1):
        frame = grab_frame_at(cap, cand.frame_idx)
        filename = f"keyframe_{rank:03d}_t_{cand.time_sec:09.3f}s.jpg"
        out_path = output_dir / filename
        width, height = save_optimized_jpeg(frame, out_path, args.max_dimension, args.jpeg_quality)
        saved_paths.append(out_path)
        metadata_frames.append(
            {
                "rank": rank,
                "filename": filename,
                "time_sec": round(float(cand.time_sec), 3),
                "frame_idx": int(cand.frame_idx),
                "scene_id": int(cand.scene_id),
                "chunk_id": int(cand.chunk_id),
                "quality_score": round(float(cand.quality_score), 4),
                "importance_score": round(float(cand.importance_score), 4),
                "select_score": round(float(cand.select_score), 4),
                "output_width": int(width),
                "output_height": int(height),
            }
        )

    cap.release()

    contact_sheet = build_contact_sheet(output_dir, saved_paths, thumb_size=args.contact_thumb_size, columns=args.contact_columns) if args.contact_sheet else None

    summary = {
        "video": str(video_path),
        "output_dir": str(output_dir),
        "duration_sec": round(float(duration_sec), 3),
        "fps": round(float(fps), 4),
        "frame_count": int(frame_count),
        "sample_fps": float(args.sample_fps),
        "sample_stride_frames": int(stride),
        "sample_count": len(samples),
        "detected_scene_cuts": len(cut_indices),
        "scene_count": len(scenes),
        "chunk_count": len(chunks),
        "budget": int(budget),
        "selected_keyframes": len(saved_paths),
        "parameters": {
            "min_scene_sec": args.min_scene_sec,
            "dynamic_max_gap_sec": args.dynamic_max_gap_sec,
            "static_max_gap_sec": args.static_max_gap_sec,
            "target_seconds_per_frame": args.target_seconds_per_frame,
            "max_frames": args.max_frames,
            "min_output_gap_sec": args.min_output_gap_sec,
            "phash_threshold": args.phash_threshold,
            "hist_threshold": args.hist_threshold,
            "max_dimension": args.max_dimension,
            "jpeg_quality": args.jpeg_quality,
        },
        "frames": metadata_frames,
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
    }

    with open(output_dir / "keyframes.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a compact set of representative keyframes from a video. "
            "The method combines adaptive scene-change detection, coverage-aware chunking, "
            "quality scoring, and near-duplicate removal."
        )
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Output folder (default: <video_stem>_keyframes next to the video)")
    parser.add_argument("--sample-fps", type=positive_float, default=2.0, help="Low-rate scan FPS used for analysis (default: 2.0)")
    parser.add_argument("--thumb-max-side", type=nonnegative_int, default=256, help="Thumbnail max side during analysis (default: 256)")
    parser.add_argument("--min-scene-sec", type=positive_float, default=1.2, help="Minimum time between scene cuts (default: 1.2)")
    parser.add_argument("--dynamic-max-gap-sec", type=positive_float, default=8.0, help="Max gap inside dynamic scenes before forcing another frame (default: 8.0)")
    parser.add_argument("--static-max-gap-sec", type=positive_float, default=60.0, help="Max gap inside static scenes before forcing another frame (default: 60.0)")
    parser.add_argument("--target-seconds-per-frame", type=positive_float, default=2.0, help="Auto-budget target spacing between output frames (default: 2.0)")
    parser.add_argument("--max-auto-frames", type=nonnegative_int, default=100, help="Upper bound when max-frames is auto (default: 100)")
    parser.add_argument("--max-frames", type=nonnegative_int, default=0, help="Hard cap on output frames; 0 means auto (default: 0)")
    parser.add_argument("--min-output-gap-sec", type=positive_float, default=2.0, help="Only dedupe frames close in time within this gap (default: 2.0)")
    parser.add_argument("--phash-threshold", type=nonnegative_int, default=4, help="Max pHash Hamming distance to treat nearby frames as duplicates (default: 8)")
    parser.add_argument("--hist-threshold", type=positive_float, default=0.03, help="Max histogram distance to treat nearby frames as duplicates (default: 0.03)")
    parser.add_argument("--max-dimension", type=nonnegative_int, default=1600, help="Resize saved JPGs so the longest side is at most this value; 0 keeps original size (default: 1600)")
    parser.add_argument("--jpeg-quality", type=nonnegative_int, default=92, help="JPEG quality for saved keyframes (default: 92)")
    parser.add_argument("--contact-sheet", action="store_true", help="Also create a contact sheet image")
    parser.add_argument("--contact-thumb-size", type=nonnegative_int, default=320, help="Thumbnail size in contact sheet (default: 320)")
    parser.add_argument("--contact-columns", type=nonnegative_int, default=4, help="Number of columns in contact sheet (default: 4)")
    return parser.parse_args()

# changed the main

def main() -> None:
    videos_dir = Path("videos")
    output_root = Path("frames")

    if not videos_dir.exists():
        raise FileNotFoundError("videos folder not found")

    video_files = sorted(videos_dir.glob("*.mp4"))

    if not video_files:
        raise RuntimeError("No .mp4 files found")

    for video_path in video_files:
        print("Processing:", video_path.name)

        class Args:
            video = str(video_path)
            output = str(output_root / video_path.stem)
            sample_fps = 8.0
            thumb_max_side = 256
            min_scene_sec = 1.2
            dynamic_max_gap_sec = 3.0
            static_max_gap_sec = 20.0
            target_seconds_per_frame = 1.0
            max_auto_frames = 100
            max_frames = 80
            min_output_gap_sec = 0.5
            phash_threshold = 12
            hist_threshold = 0.12
            max_dimension = 1600
            jpeg_quality = 92
            contact_sheet = False
            contact_thumb_size = 320
            contact_columns = 4

        out = extract_keyframes(Args)
        print("Saved to:", out)


if __name__ == "__main__":
    main()

