#!/usr/bin/env python3
"""Generate GTA-style JSON samples from surveillance video frames.

This script is batch-first for full dataset generation:
  1. prepare-batch  -> writes a JSONL file of Chat Completions requests
  2. submit-batch   -> uploads JSONL and creates an OpenAI Batch job
  3. status         -> checks a Batch job
  4. collect        -> downloads Batch results and writes final GTA-style JSON
  5. run-batch      -> prepare + submit + poll + collect

A synchronous run-sync command is included only for small debugging runs.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Any, Iterable

try:
    from PIL import Image
except Exception:  # pragma: no cover - handled at runtime with a clear error
    Image = None  # type: ignore

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_PROMPTS_DIR = Path("prompts")
DEFAULT_SYSTEM_PROMPT = DEFAULT_PROMPTS_DIR / "surveillance_system_prompt.txt"
DEFAULT_USER_PROMPT = DEFAULT_PROMPTS_DIR / "surveillance_user_prompt.txt"
DEFAULT_OUTPUT_DIR = Path("runs") / "surveillance_batch"
DEFAULT_OUTPUT_JSON = Path("surveillance_gta_dataset.json")
DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"


@dataclass(frozen=True)
class FrameSet:
    """One video represented by an ordered list of extracted frames."""

    name: str
    folder: Path
    frames: list[Path]


@dataclass(frozen=True)
class PreparedRequest:
    custom_id: str
    dataset_key: str
    video_name: str
    frame_paths: list[str]
    request: dict[str, Any]


def log(message: str) -> None:
    print(message, flush=True)


def read_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_existing_path(path_arg: str | Path, fallbacks: Iterable[str | Path] = ()) -> Path:
    candidates = [Path(path_arg), *(Path(p) for p in fallbacks)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"None of these paths exist: {rendered}")


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_json_fences(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Recovery for occasional leading/trailing prose despite JSON mode.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(cleaned[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Model response must be a JSON object")
    return obj


def normalize_tool_metadata(raw_tools: Any) -> dict[str, dict[str, Any]]:
    """Accept either {name: meta} or [meta, ...] and return {name: meta}."""
    tools: dict[str, dict[str, Any]] = {}
    if isinstance(raw_tools, dict):
        iterable = raw_tools.values()
    elif isinstance(raw_tools, list):
        iterable = raw_tools
    else:
        raise TypeError("Tool metadata must be a JSON object or list")

    for item in iterable:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            tools[name] = item
    if not tools:
        raise ValueError("No valid tools found in tool metadata")
    return tools


def select_gta_examples(gta: Any, num_examples: str, seed: int) -> list[dict[str, Any]]:
    """Select representative GTA examples from a GTA dataset JSON object/list."""
    if isinstance(gta, dict):
        examples = [gta[k] for k in sorted(gta.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
    elif isinstance(gta, list):
        examples = gta
    else:
        raise TypeError("GTA dataset must be a JSON object or list")

    examples = [x for x in examples if isinstance(x, dict)]
    if not examples:
        raise ValueError("No GTA examples found")

    if str(num_examples).lower() == "all":
        return examples

    n = int(num_examples)
    if n <= 0:
        return []
    if len(examples) <= n:
        return examples

    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(examples)), n))
    return [examples[i] for i in idxs]


def image_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "image/jpeg"


def encode_image_for_request(path: Path, max_side: int, jpeg_quality: int) -> str:
    """Return a data URL. Optionally resize/re-encode to keep Batch JSONL below limits."""
    if max_side <= 0 and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{image_mime(path)};base64,{b64}"

    if Image is None:
        raise RuntimeError("Pillow is required for image resizing. Install with: pip install pillow")

    with Image.open(path) as img:
        img = img.convert("RGB")
        if max_side > 0:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def sorted_image_files(folder: Path) -> list[Path]:
    files = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.lower().startswith("contact_sheet")
    ]
    return sorted(files, key=lambda p: p.name)


def discover_frame_sets(frames_dir: Path) -> list[FrameSet]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    direct_frames = sorted_image_files(frames_dir)
    if direct_frames:
        return [FrameSet(name=frames_dir.name, folder=frames_dir, frames=direct_frames)]

    frame_sets: list[FrameSet] = []
    for child in sorted(frames_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        frames = sorted_image_files(child)
        if frames:
            frame_sets.append(FrameSet(name=child.name, folder=child, frames=frames))
    if not frame_sets:
        raise RuntimeError(f"No frame folders found in {frames_dir}")
    return frame_sets


def uniformly_limit_frames(frames: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    indexes = sorted({round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [frames[i] for i in indexes]


def path_for_json(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_keyframe_metadata(folder: Path) -> dict[str, dict[str, Any]]:
    metadata_path = folder / "keyframes.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata = load_json(metadata_path)
    except Exception:
        return {}
    frames = metadata.get("frames", []) if isinstance(metadata, dict) else []
    by_name: dict[str, dict[str, Any]] = {}
    for item in frames:
        if isinstance(item, dict) and isinstance(item.get("filename"), str):
            by_name[item["filename"]] = item
    return by_name


def build_files_section(frame_paths: list[str]) -> list[dict[str, Any]]:
    return [{"type": "image", "path": p, "url": ""} for p in frame_paths]


def build_frame_manifest(frame_set: FrameSet, chosen_frames: list[Path], project_root: Path) -> list[dict[str, Any]]:
    metadata_by_name = load_keyframe_metadata(frame_set.folder)
    manifest: list[dict[str, Any]] = []
    for order, frame in enumerate(chosen_frames, start=1):
        meta = metadata_by_name.get(frame.name, {})
        item: dict[str, Any] = {
            "order": order,
            "type": "image",
            "path": path_for_json(frame, project_root),
            "url": "",
        }
        for key in ("time_sec", "frame_idx", "scene_id", "chunk_id"):
            if key in meta:
                item[key] = meta[key]
        manifest.append(item)
    return manifest


def render_user_prompt(template_path: Path, gta_examples: list[dict[str, Any]], tools_meta: dict[str, dict[str, Any]], frame_manifest: list[dict[str, Any]]) -> str:
    template = Template(read_text(template_path))
    return template.safe_substitute(
        GTA_EXAMPLES_JSON=json.dumps(gta_examples, indent=2, ensure_ascii=False),
        TOOL_METADATA_JSON=json.dumps(tools_meta, indent=2, ensure_ascii=False),
        FRAME_MANIFEST_JSON=json.dumps(frame_manifest, indent=2, ensure_ascii=False),
    )


def build_chat_request(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    frame_files: list[Path],
    image_detail: str,
    image_max_side: int,
    jpeg_quality: int,
    max_tokens: int,
    temperature: float | None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for frame in frame_files:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_for_request(frame, image_max_side, jpeg_quality),
                    "detail": image_detail,
                },
            }
        )

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": DEFAULT_BATCH_ENDPOINT,
        "body": body,
    }


def prepare_requests(args: argparse.Namespace) -> list[PreparedRequest]:
    project_root = Path(args.project_root).resolve()
    frames_dir = resolve_existing_path(args.frames_dir).resolve()
    tools_path = resolve_existing_path(args.tools_file, ["toolmeta.JSON", "toolsmeta.json", "toolmeta.json"])
    gta_path = resolve_existing_path(args.gta_file, ["GTA dataset.json", "GTA_dataset.json"])
    system_prompt_path = resolve_existing_path(args.system_prompt)
    user_prompt_path = resolve_existing_path(args.user_prompt)

    raw_tools = load_json(tools_path)
    tools_meta = normalize_tool_metadata(raw_tools)
    gta_examples = select_gta_examples(load_json(gta_path), args.num_gta_examples, args.seed)
    system_prompt = read_text(system_prompt_path)

    frame_sets = discover_frame_sets(frames_dir)
    prepared: list[PreparedRequest] = []

    for idx, frame_set in enumerate(frame_sets):
        chosen_frames = uniformly_limit_frames(frame_set.frames, args.max_frames_per_video)
        frame_manifest = build_frame_manifest(frame_set, chosen_frames, project_root)
        user_prompt = render_user_prompt(user_prompt_path, gta_examples, tools_meta, frame_manifest)
        custom_id = f"video-{idx:06d}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', frame_set.name)}"
        request = build_chat_request(
            custom_id=custom_id,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            frame_files=chosen_frames,
            image_detail=args.image_detail,
            image_max_side=args.batch_image_max_side,
            jpeg_quality=args.batch_jpeg_quality,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        prepared.append(
            PreparedRequest(
                custom_id=custom_id,
                dataset_key=str(idx),
                video_name=frame_set.name,
                frame_paths=[item["path"] for item in frame_manifest],
                request=request,
            )
        )
    return prepared


def write_batch_files(prepared: list[PreparedRequest], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "requests.jsonl"
    manifest_path = output_dir / "manifest.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in prepared:
            f.write(json.dumps(item.request, ensure_ascii=False) + "\n")

    manifest = {
        "endpoint": DEFAULT_BATCH_ENDPOINT,
        "requests_jsonl": str(jsonl_path),
        "items": [
            {
                "custom_id": item.custom_id,
                "dataset_key": item.dataset_key,
                "video_name": item.video_name,
                "frame_paths": item.frame_paths,
            }
            for item in prepared
        ],
    }
    write_json(manifest_path, manifest)
    return jsonl_path, manifest_path


def command_prepare_batch(args: argparse.Namespace) -> None:
    prepared = prepare_requests(args)
    jsonl_path, manifest_path = write_batch_files(prepared, Path(args.output_dir))
    size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    log(f"Prepared {len(prepared)} requests")
    log(f"JSONL: {jsonl_path} ({size_mb:.2f} MB)")
    log(f"Manifest: {manifest_path}")
    if size_mb > 190:
        log("WARNING: Batch JSONL is close to or above the 200 MB upload limit. Reduce --max-frames-per-video or --batch-image-max-side.")


def read_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    key_file = Path("openai_key.txt")
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise RuntimeError("Set OPENAI_API_KEY or create openai_key.txt containing your API key")


def get_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc
    return OpenAI(api_key=read_api_key())


def command_submit_batch(args: argparse.Namespace) -> None:
    client = get_client()
    input_jsonl = resolve_existing_path(args.input_jsonl)
    with input_jsonl.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=DEFAULT_BATCH_ENDPOINT,
        completion_window="24h",
        metadata={"project": "surveillance_gta", "source": "prompt_py"},
    )
    log(f"Uploaded file_id: {uploaded.id}")
    log(f"Batch id: {batch.id}")
    if args.batch_id_file:
        Path(args.batch_id_file).write_text(batch.id + "\n", encoding="utf-8")
        log(f"Saved batch id to: {args.batch_id_file}")


def command_status(args: argparse.Namespace) -> None:
    client = get_client()
    batch = client.batches.retrieve(args.batch_id)
    log(json.dumps(batch.model_dump(), indent=2, ensure_ascii=False))


def response_text_from_chat_body(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Batch response body has no choices")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Batch response message content is not text")
    return content


def load_batch_output_lines(client: Any, batch: Any, output_path: Path | None = None) -> list[dict[str, Any]]:
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("Batch has no output_file_id yet. Check status first.")

    content = client.files.content(output_file_id)
    if hasattr(content, "text"):
        text = content.text
    elif hasattr(content, "read"):
        raw = content.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    else:
        text = str(content)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    lines = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse output JSONL line {line_no}: {exc}") from exc
    return lines


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(manifest_path)
    items = manifest.get("items", []) if isinstance(manifest, dict) else []
    mapping: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("custom_id"), str):
            mapping[item["custom_id"]] = item
    if not mapping:
        raise ValueError(f"No manifest items found in {manifest_path}")
    return mapping


def extract_used_tool_names(sample: dict[str, Any]) -> list[str]:
    names: list[str] = []

    def add(name: Any) -> None:
        if isinstance(name, str) and name and name not in names:
            names.append(name)

    tools = sample.get("tools", [])
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                add(tool.get("name"))
            elif isinstance(tool, str):
                add(tool)

    dialogs = sample.get("dialogs", [])
    if isinstance(dialogs, list):
        for turn in dialogs:
            if not isinstance(turn, dict):
                continue
            if turn.get("role") == "tool":
                add(turn.get("name"))
            tool_calls = turn.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function", {})
                    if isinstance(fn, dict):
                        add(fn.get("name"))
    return names


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return normalized
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function", {})
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name") or call.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = fn.get("arguments", call.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"input": args}
        normalized.append({"type": "function", "function": {"name": name, "arguments": args}})
    return normalized


def normalize_tool_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        if "type" in content and "content" in content:
            return content
        return {"type": "text", "content": json.dumps(content, ensure_ascii=False)}
    if content is None:
        return {"type": "text", "content": ""}
    return {"type": "text", "content": str(content)}


def normalize_dialogs(dialogs: Any) -> list[dict[str, Any]]:
    if not isinstance(dialogs, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for turn in dialogs:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if role == "user":
            cleaned.append({"role": "user", "content": str(turn.get("content", ""))})
        elif role == "assistant":
            tool_calls = normalize_tool_calls(turn.get("tool_calls"))
            if tool_calls:
                cleaned_turn = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                    "thought": str(turn.get("thought", "")),
                }
            else:
                cleaned_turn = {"role": "assistant", "content": str(turn.get("content", ""))}
            cleaned.append(cleaned_turn)
        elif role == "tool":
            name = turn.get("name")
            if not isinstance(name, str) or not name:
                continue
            cleaned.append({"role": "tool", "name": name, "content": normalize_tool_content(turn.get("content"))})
    return cleaned


def normalize_gt_answer(gt_answer: Any, final_answer: str) -> dict[str, Any]:
    answer = final_answer
    if isinstance(gt_answer, dict):
        whitelist = gt_answer.get("whitelist")
        if isinstance(whitelist, list) and whitelist:
            first_group = whitelist[0]
            if isinstance(first_group, list) and first_group and first_group[0] is not None:
                answer = str(first_group[0])
            elif first_group is not None:
                answer = str(first_group)
    if not answer:
        answer = ""
    return {"whitelist": [[answer]], "blacklist": None}


def normalize_sample(sample: dict[str, Any], frame_paths: list[str], tools_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Force the output into the exact GTA top-level structure."""
    dialogs = normalize_dialogs(sample.get("dialogs", []))

    # Ensure a final assistant answer exists and matches gt_answer if possible.
    final_answer = ""
    for turn in reversed(dialogs):
        if turn.get("role") == "assistant" and "content" in turn:
            final_answer = str(turn.get("content", ""))
            break
    gt_answer = normalize_gt_answer(sample.get("gt_answer"), final_answer)
    normalized_final = gt_answer["whitelist"][0][0]

    if not dialogs or dialogs[0].get("role") != "user":
        dialogs.insert(0, {"role": "user", "content": "What observable surveillance event changes across these frames?"})

    if not dialogs or dialogs[-1].get("role") != "assistant" or "content" not in dialogs[-1]:
        dialogs.append({"role": "assistant", "content": normalized_final})
    else:
        dialogs[-1] = {"role": "assistant", "content": normalized_final}

    used_names = extract_used_tool_names({**sample, "dialogs": dialogs})
    used_tool_defs = [tools_meta[name] for name in used_names if name in tools_meta]

    # If the model emitted unknown tool names, keep only known metadata in tools but let validation report unknowns.
    return {
        "tools": used_tool_defs,
        "files": build_files_section(frame_paths),
        "dialogs": dialogs,
        "gt_answer": gt_answer,
    }


def validate_sample(sample: dict[str, Any], frame_paths: list[str], tools_meta: dict[str, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if list(sample.keys()) != ["tools", "files", "dialogs", "gt_answer"]:
        warnings.append("top-level keys/order differ from GTA structure")

    if sample.get("files") != build_files_section(frame_paths):
        warnings.append("files section does not exactly match frame manifest")

    used_names = extract_used_tool_names(sample)
    unknown = [name for name in used_names if name not in tools_meta]
    if unknown:
        warnings.append(f"unknown tools used: {unknown}")

    dialogs = sample.get("dialogs")
    if not isinstance(dialogs, list) or not dialogs:
        warnings.append("dialogs missing or empty")
    else:
        if dialogs[0].get("role") != "user":
            warnings.append("first dialog turn is not user")
        if dialogs[-1].get("role") != "assistant" or "content" not in dialogs[-1]:
            warnings.append("last dialog turn is not assistant final content")

    gt = sample.get("gt_answer")
    if not isinstance(gt, dict) or "whitelist" not in gt or gt.get("blacklist", None) is not None:
        warnings.append("gt_answer is not GTA-style")
    return warnings


def output_line_to_sample(line: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    custom_id = line.get("custom_id")
    if not isinstance(custom_id, str):
        return "", None, "missing custom_id"
    if line.get("error"):
        return custom_id, None, json.dumps(line.get("error"), ensure_ascii=False)
    response = line.get("response")
    if not isinstance(response, dict):
        return custom_id, None, "missing response object"
    if response.get("status_code") != 200:
        return custom_id, None, f"status_code={response.get('status_code')}: {response.get('body')}"
    body = response.get("body")
    if not isinstance(body, dict):
        return custom_id, None, "missing response body"
    try:
        return custom_id, parse_json_object(response_text_from_chat_body(body)), None
    except Exception as exc:
        return custom_id, None, f"parse error: {exc}"


def collect_from_lines(
    lines: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    tools_meta: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset: dict[str, Any] = {}
    report = {"ok": 0, "failed": 0, "warnings": {}, "errors": {}}

    for line in lines:
        custom_id, raw_sample, error = output_line_to_sample(line)
        manifest_item = manifest_by_id.get(custom_id)
        if manifest_item is None:
            report["failed"] += 1
            report["errors"][custom_id or f"line_{report['failed']}"] = "custom_id not found in manifest"
            continue
        dataset_key = str(manifest_item["dataset_key"])
        frame_paths = list(manifest_item["frame_paths"])
        if error or raw_sample is None:
            report["failed"] += 1
            report["errors"][custom_id] = error or "unknown error"
            continue
        normalized = normalize_sample(raw_sample, frame_paths, tools_meta)
        warnings = validate_sample(normalized, frame_paths, tools_meta)
        if warnings:
            report["warnings"][custom_id] = warnings
        dataset[dataset_key] = normalized
        report["ok"] += 1

    dataset = dict(sorted(dataset.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))
    return dataset, report


def command_collect(args: argparse.Namespace) -> None:
    client = get_client()
    manifest_by_id = load_manifest(resolve_existing_path(args.manifest))
    tools_path = resolve_existing_path(args.tools_file, ["toolmeta.JSON", "toolsmeta.json", "toolmeta.json"])
    tools_meta = normalize_tool_metadata(load_json(tools_path))
    batch = client.batches.retrieve(args.batch_id)

    raw_output_path = Path(args.raw_output_jsonl) if args.raw_output_jsonl else None
    lines = load_batch_output_lines(client, batch, raw_output_path)
    dataset, report = collect_from_lines(lines, manifest_by_id, tools_meta)

    output_file = Path(args.output_file)
    write_json(output_file, dataset)
    report_path = Path(args.report_file) if args.report_file else output_file.with_suffix(".report.json")
    write_json(report_path, report)
    log(f"Saved dataset: {output_file}")
    log(f"Saved report: {report_path}")
    log(f"OK: {report['ok']} | Failed: {report['failed']} | Warnings: {len(report['warnings'])}")


def command_run_batch(args: argparse.Namespace) -> None:
    prepared = prepare_requests(args)
    output_dir = Path(args.output_dir)
    jsonl_path, manifest_path = write_batch_files(prepared, output_dir)
    size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    log(f"Prepared {len(prepared)} requests at {jsonl_path} ({size_mb:.2f} MB)")
    if size_mb > 190:
        log("WARNING: Batch JSONL is close to or above the 200 MB upload limit. Reduce frame count or image size if upload fails.")

    client = get_client()
    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=DEFAULT_BATCH_ENDPOINT,
        completion_window="24h",
        metadata={"project": "surveillance_gta", "source": "prompt_py_run_batch"},
    )
    (output_dir / "batch_id.txt").write_text(batch.id + "\n", encoding="utf-8")
    log(f"Batch id: {batch.id}")

    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch.id)
        status = getattr(batch, "status", "unknown")
        counts = getattr(batch, "request_counts", None)
        counts_dump = counts.model_dump() if hasattr(counts, "model_dump") else counts
        log(f"Batch status: {status}; counts={counts_dump}")
        if status in terminal:
            break
        time.sleep(args.poll_seconds)

    if getattr(batch, "status", None) != "completed":
        raise RuntimeError(f"Batch did not complete successfully. Final status: {getattr(batch, 'status', None)}")

    manifest_by_id = load_manifest(manifest_path)
    tools_path = resolve_existing_path(args.tools_file, ["toolmeta.JSON", "toolsmeta.json", "toolmeta.json"])
    tools_meta = normalize_tool_metadata(load_json(tools_path))
    raw_output_path = output_dir / "batch_output.jsonl"
    lines = load_batch_output_lines(client, batch, raw_output_path)
    dataset, report = collect_from_lines(lines, manifest_by_id, tools_meta)
    write_json(Path(args.output_file), dataset)
    write_json(output_dir / "report.json", report)
    log(f"Saved final JSON: {args.output_file}")


def command_run_sync(args: argparse.Namespace) -> None:
    """Small synchronous debug run. Full generation should use run-batch."""
    prepared = prepare_requests(args)
    if args.limit > 0:
        prepared = prepared[: args.limit]
    tools_path = resolve_existing_path(args.tools_file, ["toolmeta.JSON", "toolsmeta.json", "toolmeta.json"])
    tools_meta = normalize_tool_metadata(load_json(tools_path))
    client = get_client()

    dataset: dict[str, Any] = {}
    report = {"ok": 0, "failed": 0, "warnings": {}, "errors": {}}
    for item in prepared:
        log(f"Processing {item.video_name} ({item.custom_id})")
        try:
            body = item.request["body"]
            response = client.chat.completions.create(**body)
            text = response.choices[0].message.content or "{}"
            raw_sample = parse_json_object(text)
            normalized = normalize_sample(raw_sample, item.frame_paths, tools_meta)
            warnings = validate_sample(normalized, item.frame_paths, tools_meta)
            if warnings:
                report["warnings"][item.custom_id] = warnings
            dataset[item.dataset_key] = normalized
            report["ok"] += 1
        except Exception as exc:
            report["failed"] += 1
            report["errors"][item.custom_id] = str(exc)
            log(f"Failed {item.custom_id}: {exc}")

    dataset = dict(sorted(dataset.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))
    write_json(Path(args.output_file), dataset)
    write_json(Path(args.output_file).with_suffix(".report.json"), report)
    log(f"Saved final JSON: {args.output_file}")


def command_validate(args: argparse.Namespace) -> None:
    dataset = load_json(resolve_existing_path(args.dataset))
    tools_path = resolve_existing_path(args.tools_file, ["toolmeta.JSON", "toolsmeta.json", "toolmeta.json"])
    tools_meta = normalize_tool_metadata(load_json(tools_path))
    manifest_by_id: dict[str, dict[str, Any]] = {}
    if args.manifest:
        manifest_by_id = load_manifest(resolve_existing_path(args.manifest))
    frame_paths_by_key = {str(v["dataset_key"]): list(v["frame_paths"]) for v in manifest_by_id.values()}

    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be a JSON object keyed by sample id")
    all_warnings: dict[str, list[str]] = {}
    for key, sample in dataset.items():
        if not isinstance(sample, dict):
            all_warnings[str(key)] = ["sample is not an object"]
            continue
        frame_paths = frame_paths_by_key.get(str(key))
        if frame_paths is None:
            files = sample.get("files", [])
            frame_paths = [f.get("path") for f in files if isinstance(f, dict) and isinstance(f.get("path"), str)]
        warnings = validate_sample(sample, frame_paths, tools_meta)
        if warnings:
            all_warnings[str(key)] = warnings
    if all_warnings:
        log(json.dumps(all_warnings, indent=2, ensure_ascii=False))
        raise SystemExit(1)
    log(f"Validated {len(dataset)} samples successfully")


def add_common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=".", help="Root used to make frame paths relative in output JSON")
    parser.add_argument("--frames-dir", default="frames", help="Folder containing per-video frame folders")
    parser.add_argument("--tools-file", default="toolmeta.JSON", help="Tool metadata JSON file")
    parser.add_argument("--gta-file", default="GTA dataset.json", help="Reference GTA dataset JSON file")
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT), help="System prompt text file")
    parser.add_argument("--user-prompt", default=str(DEFAULT_USER_PROMPT), help="User prompt template text file")
    parser.add_argument("--num-gta-examples", default="5", help="Number of GTA examples to include, or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="Seed for GTA example selection")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"), help="OpenAI model for generation")
    parser.add_argument("--max-frames-per-video", type=int, default=24, help="Uniformly cap attached frames per video; 0 means all")
    parser.add_argument("--batch-image-max-side", type=int, default=1024, help="Resize image payloads for API requests; 0 keeps original bytes")
    parser.add_argument("--batch-jpeg-quality", type=int, default=85, help="JPEG quality for resized API image payloads")
    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default="low", help="Vision detail setting")
    parser.add_argument("--max-tokens", type=int, default=6000, help="Maximum output tokens per sample")
    parser.add_argument("--temperature", type=float, default=0.4, help="Generation temperature; use -1 to omit")


def coerce_temperature(args: argparse.Namespace) -> None:
    if hasattr(args, "temperature") and args.temperature is not None and args.temperature < 0:
        args.temperature = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Surveillance GTA-style dataset generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-batch", help="Create Batch API JSONL and manifest files")
    add_common_generation_args(p)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for requests.jsonl and manifest.json")
    p.set_defaults(func=command_prepare_batch)

    p = sub.add_parser("submit-batch", help="Upload a prepared JSONL file and create a Batch job")
    p.add_argument("--input-jsonl", default=str(DEFAULT_OUTPUT_DIR / "requests.jsonl"), help="Prepared JSONL file")
    p.add_argument("--batch-id-file", default=str(DEFAULT_OUTPUT_DIR / "batch_id.txt"), help="Where to save the created batch id")
    p.set_defaults(func=command_submit_batch)

    p = sub.add_parser("status", help="Print Batch job status")
    p.add_argument("--batch-id", required=True, help="Batch id")
    p.set_defaults(func=command_status)

    p = sub.add_parser("collect", help="Collect Batch output and write final GTA-style JSON")
    p.add_argument("--batch-id", required=True, help="Completed Batch id")
    p.add_argument("--manifest", default=str(DEFAULT_OUTPUT_DIR / "manifest.json"), help="Manifest created by prepare-batch")
    p.add_argument("--tools-file", default="toolmeta.JSON", help="Tool metadata JSON file")
    p.add_argument("--output-file", default=str(DEFAULT_OUTPUT_JSON), help="Final dataset JSON")
    p.add_argument("--raw-output-jsonl", default=str(DEFAULT_OUTPUT_DIR / "batch_output.jsonl"), help="Where to save raw Batch JSONL output")
    p.add_argument("--report-file", default=str(DEFAULT_OUTPUT_DIR / "report.json"), help="Validation/collection report JSON")
    p.set_defaults(func=command_collect)

    p = sub.add_parser("run-batch", help="Prepare, submit, poll, and collect using Batch API")
    add_common_generation_args(p)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Run directory")
    p.add_argument("--output-file", default=str(DEFAULT_OUTPUT_JSON), help="Final dataset JSON")
    p.add_argument("--poll-seconds", type=int, default=60, help="Seconds between Batch status checks")
    p.set_defaults(func=command_run_batch)

    p = sub.add_parser("run-sync", help="Synchronous debug run for a small number of videos")
    add_common_generation_args(p)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Temporary run directory")
    p.add_argument("--output-file", default=str(DEFAULT_OUTPUT_JSON), help="Final dataset JSON")
    p.add_argument("--limit", type=int, default=1, help="Number of videos to process synchronously; 0 means all")
    p.set_defaults(func=command_run_sync)

    p = sub.add_parser("validate", help="Validate a generated GTA-style JSON file")
    p.add_argument("--dataset", default=str(DEFAULT_OUTPUT_JSON), help="Dataset JSON to validate")
    p.add_argument("--tools-file", default="toolmeta.JSON", help="Tool metadata JSON file")
    p.add_argument("--manifest", default="", help="Optional manifest.json from prepare-batch")
    p.set_defaults(func=command_validate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    coerce_temperature(args)
    args.func(args)


if __name__ == "__main__":
    main()
