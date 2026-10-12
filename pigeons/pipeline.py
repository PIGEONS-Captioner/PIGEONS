#!/usr/bin/env python3
"""PIGEONS two-pass audiovisual captioning with a Swift endpoint.

1. timestamp-stage repository patch from the current clip and existing repo
2. full-frame review that returns either bbox or corrected timestamp
3. crop review that accepts references or returns corrected crop parameters
4. repeat failed visual detections up to ``--max-review-rounds``

After the gallery is fixed, captions retain the checkpoint-compatible summary protocol.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


from . import gallery, media
from .media import split_video_with_retry


IMAGE_FACTOR = 32
MAX_PIXELS = 256 * IMAGE_FACTOR * IMAGE_FACTOR
JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.I | re.S)
SUBJECT_ID_SEARCH_RE = re.compile(r"\b[A-Z]+_[A-Z]\b")
LEGACY_SUBJECT_ID_SEARCH_RE = re.compile(r"<[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+>")

SYSTEM_PROMPT = gallery.GALLERY_SYSTEM_PROMPT
CAPTION_SYSTEM_PROMPT = (
    "Generate a detailed chronological audio-visual caption and updated "
    "cumulative structured summary as JSON. Use only confirmed subject IDs "
    "in the TYPE_LETTER namespace without a prefix, angle brackets, or digits, "
    "and follow the user's coverage and accuracy requirements."
)
EMPTY_SUMMARY = {"backgrounds": [], "established_facts": []}
BACKEND_UNHEALTHY_MARKERS = (
    "enginecore encountered an issue",
    "enginedeaderror",
    "engine dead",
    "connection refused",
    "connection reset",
    "connection aborted",
    "connectionerror",
    "remoteprotocolerror",
    "server disconnected",
    "service unavailable",
    "internal server error",
    "http 500",
    "http 503",
    "status code: 500",
    "status code: 503",
    "broken pipe",
)


class ModelJSONDecodeError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class PartialRepositoryBuildError(RuntimeError):
    def __init__(
        self,
        message: str,
        repository: dict[str, Any],
        clip_result: dict[str, Any],
        cause: Exception,
    ) -> None:
        super().__init__(message)
        self.repository = deepcopy(repository)
        self.clip_result = deepcopy(clip_result)
        self.cause = cause


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PIGEONS identity gallery and generate audiovisual captions "
            "for one or more videos."
        )
    )
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--clip-dir", type=Path, default=None, help="Clip cache directory (default: OUTPUT_DIR/clips)")
    parser.add_argument(
        "--dataset-label",
        default=None,
        help=(
            "Human-readable dataset label for output directory names. "
            "If omitted, infer from paths containing Datasets/<name>/..."
        ),
    )
    parser.add_argument("--clip-seconds", type=int, default=120)
    parser.add_argument("--max-clip-seconds", type=int, default=125)
    generation = parser.add_argument_group(
        "Generation options",
        "Sent with every gallery and caption request; override server defaults.",
    )
    generation.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Maximum generated tokens per request (default: 8192)",
    )
    generation.add_argument(
        "--temperature", type=float, default=0.0,
        help="Request temperature (default: 0.0)",
    )
    generation.add_argument(
        "--top_k", type=int, default=1, help="Request top-k sampling (default: 1)",
    )
    generation.add_argument(
        "--top_p", type=float, default=0.001, help="Request top-p sampling (default: 0.001)",
    )
    generation.add_argument(
        "--seed", type=int, default=42, help="Per-request seed (default: 42)",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--validation-retries", type=int, default=3)
    parser.add_argument("--max-review-rounds", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("INFER_WORKERS", "1")),
        help="Number of videos to infer concurrently. Clips within one video stay sequential.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preserve-clips-on-rerun",
        action="store_true",
        help=(
            "When rerunning an incomplete or overwritten video, keep existing split "
            "clips so a shared clip directory can be reused across experiments."
        ),
    )
    parser.add_argument(
        "--fail-fast-on-backend-error",
        action="store_true",
        help=(
            "Stop the batch quickly when a vLLM/backend health error is detected. "
            "This is intended for use with an external supervisor that restarts the server."
        ),
    )
    return parser.parse_args()


def iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def is_backend_unhealthy_error(exc: BaseException) -> bool:
    for item in iter_exception_chain(exc):
        text = f"{type(item).__name__}: {item}".lower()
        if any(marker in text for marker in BACKEND_UNHEALTHY_MARKERS):
            return True
    return False


def matched_backend_markers(exc: BaseException) -> list[str]:
    text = "\n".join(
        f"{type(item).__name__}: {item}".lower() for item in iter_exception_chain(exc)
    )
    return [marker for marker in BACKEND_UNHEALTHY_MARKERS if marker in text]


def exception_chain_payload(exc: BaseException) -> list[dict[str, str]]:
    return [
        {"type": type(item).__name__, "message": str(item)}
        for item in iter_exception_chain(exc)
    ]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_backend_unhealthy_event(
    args: argparse.Namespace,
    *,
    video: Path,
    exc: BaseException,
    worker_count: int,
    completed_count: int,
    failure_count: int,
) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": "backend_unhealthy",
        "video": str(video),
        "host": args.host,
        "port": args.port,
        "workers": worker_count,
        "completed_count": completed_count,
        "failure_count": failure_count,
        "matched_markers": matched_backend_markers(exc),
        "exception_chain": exception_chain_payload(exc),
    }
    append_jsonl(args.output_dir / "backend_unhealthy_events.jsonl", record)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def elapsed_seconds(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def parse_json_response(text: str) -> dict[str, Any]:
    content = text.strip()
    candidates: list[str] = []
    matches = list(JSON_BLOCK_RE.finditer(content))
    if matches:
        candidates.append(matches[-1].group(1).strip())
    left, right = content.find("{"), content.rfind("}")
    if left >= 0 and right > left:
        candidates.append(content[left : right + 1])
    candidates.append(content)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ModelJSONDecodeError("Model did not return a JSON object", content)


def parse_caption_json_response(text: str) -> dict[str, Any]:
    content = text.strip()
    match = JSON_BLOCK_RE.search(content)
    candidates = [match.group(1).strip()] if match else []
    candidates.append(content)
    left, right = content.find("{"), content.rfind("}")
    if left >= 0 and right > left:
        candidates.append(content[left : right + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    preview = content[:500]
    raise ModelJSONDecodeError(
        f"Model did not return a JSON object: {preview}", content
    )


def extract_json_with_analysis(text: str) -> dict[str, Any]:
    try:
        parsed, _analysis = gallery.extract_final_json_object(text)
        return parsed
    except Exception:
        return parse_json_response(text)


def make_message(
    role: str,
    text: str,
    videos: list[str] | None = None,
    images: list[str] | None = None,
    fps: float = 2.0,
    max_frames: int = 240,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for video in videos or []:
        content.append(
            {
                "type": "video",
                "video": video,
                "max_pixels": MAX_PIXELS,
                "fps": fps,
                "max_frames": max_frames,
            }
        )
    for image in images or []:
        content.append({"type": "image", "image": image, "max_pixels": MAX_PIXELS})
    content.append({"type": "text", "text": text})
    return {"role": role, "content": content}


def build_request(conversation: list[dict[str, Any]]) -> Any:
    messages: list[dict[str, str]] = []
    videos: list[str] = []
    images: list[str] = []
    audios: list[str] = []
    video_settings: list[dict[str, Any]] = []
    for message in conversation:
        content = message["content"]
        if isinstance(content, str):
            text = content
            video_count = image_count = audio_count = 0
        else:
            text_parts = []
            video_count = image_count = audio_count = 0
            for item in content:
                item_type = item["type"]
                if item_type == "text":
                    text_parts.append(item["text"])
                elif item_type == "video":
                    videos.append(item["video"])
                    video_settings.append({key: item[key] for key in ("fps", "max_frames", "max_pixels") if key in item})
                    video_count += 1
                elif item_type == "image":
                    images.append(item["image"])
                    image_count += 1
                elif item_type == "audio":
                    audios.append(item["audio"])
                    audio_count += 1
                else:
                    raise ValueError(f"Unsupported media type: {item_type}")
            text = "".join(text_parts)
        if text.count("<video>") < video_count:
            text = "<video>\n" * (video_count - text.count("<video>")) + text
        if text.count("<image>") < image_count:
            text += "\n<image>" * (image_count - text.count("<image>"))
        if text.count("<audio>") < audio_count:
            text += "\n<audio>" * (audio_count - text.count("<audio>"))
        messages.append({"role": message["role"], "content": text})
    from swift import InferRequest

    return InferRequest(
        messages=messages,
        videos=videos,
        images=images,
        audios=audios,
        objects={
            "source": "deploy",
            "video_paths": videos,
            "video_settings": video_settings,
            # Swift normalizes bbox whenever images and objects are non-empty.
            "bbox": [],
        },
    )


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        from swift import InferClient, InferStats, RequestConfig

        self.client = InferClient(host=args.host, port=args.port)
        # Explicit request settings take precedence over Swift's server defaults.
        self.config = RequestConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        self.metric = InferStats()

    def generate(self, conversation: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        response = self.client.infer(
            [build_request(conversation)], self.config, metrics=[self.metric]
        )[0]
        raw = response.choices[0].message.content.strip()
        return extract_json_with_analysis(raw), raw

    def generate_caption(
        self, conversation: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str]:
        response = self.client.infer(
            [build_request(conversation)], self.config, metrics=[self.metric]
        )[0]
        raw = response.choices[0].message.content.strip()
        return parse_caption_json_response(raw), raw


def generate_validated(
    model: Model,
    conversation: list[dict[str, Any]],
    validator: Callable[[dict[str, Any]], Any],
    retries: int,
    attempt_logger: Callable[
        [
            int,
            list[dict[str, Any]],
            dict[str, Any] | None,
            str | None,
            Exception | None,
            bool,
            float,
        ],
        None,
    ]
    | None = None,
) -> tuple[dict[str, Any], str]:
    if retries < 1:
        raise ValueError("--validation-retries must be >= 1")
    last_error: Exception | None = None
    for attempt in range(retries):
        raw: str | None = None
        payload: dict[str, Any] | None = None
        attempt_start = time.perf_counter()
        try:
            payload, raw = model.generate(conversation)
            validator(payload)
            duration_s = elapsed_seconds(attempt_start)
            if attempt_logger is not None:
                attempt_logger(
                    attempt, conversation, payload, raw, None, True, duration_s
                )
            return payload, raw
        except Exception as exc:
            duration_s = elapsed_seconds(attempt_start)
            last_error = exc
            raw = getattr(exc, "raw_response", raw)
            if attempt_logger is not None:
                attempt_logger(
                    attempt, conversation, payload, raw, exc, False, duration_s
                )
            message = str(exc)
            if (
                "maximum model length" in message
                or "decoder prompt" in message
                or "max_model_len" in message
            ):
                raise RuntimeError(f"VLLM backend request failed: {message}") from exc
            if raw is None and (
                "EngineCore encountered an issue" in message
                or "Connection refused" in message
                or "ConnectionError" in type(exc).__name__
                or is_backend_unhealthy_error(exc)
            ):
                raise RuntimeError(f"VLLM backend request failed: {message}") from exc
            if attempt + 1 < retries and raw is not None:
                conversation.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "The previous JSON response failed validation: "
                                f"{exc}. Return concise analysis if useful, then end "
                                "with one complete corrected JSON object matching the "
                                "same requested schema."
                            ),
                        },
                    ]
                )
    raise RuntimeError(
        f"Model response failed validation after {retries} attempts: {last_error}"
    ) from last_error


def generate_turn(
    model: Model,
    conversation: list[dict[str, Any]],
    user_message: dict[str, Any],
    validator: Callable[[dict[str, Any]], Any],
    retries: int,
    *,
    stage: str,
    round_index: int | None,
    interactions: list[dict[str, Any]],
    interaction_output_dir: Path,
    clip_index: int,
    clip: str,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    request_conversation = [*conversation, user_message]

    def log_attempt(
        attempt_index: int,
        attempt_conversation: list[dict[str, Any]],
        attempt_payload: dict[str, Any] | None,
        attempt_raw: str | None,
        error: Exception | None,
        valid: bool,
        duration_s: float,
    ) -> None:
        append_interaction(
            interactions,
            stage,
            round_index,
            attempt_conversation,
            attempt_payload,
            attempt_raw,
            attempt_index=attempt_index,
            status="ok" if valid else "failed_validation",
            error=error,
            duration_s=duration_s,
        )
        dump_model_interactions(
            interaction_output_dir,
            clip_index,
            clip,
            interactions,
        )

    payload, raw = generate_validated(
        model,
        request_conversation,
        validator,
        retries,
        attempt_logger=log_attempt,
    )
    conversation.extend(
        [
            user_message,
            {"role": "assistant", "content": raw},
        ]
    )
    return payload, raw, request_conversation


def count_conversation_images(conversation: list[dict[str, Any]]) -> int:
    count = 0
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            count += sum(1 for item in content if item.get("type") == "image")
    return count


def serialize_request_conversation(
    request_conversation: list[dict[str, Any]],
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    videos: list[str] = []
    images: list[str] = []
    audios: list[str] = []
    video_settings: list[dict[str, Any]] = []
    for message in request_conversation:
        content = message["content"]
        if isinstance(content, str):
            text = content
        else:
            text_parts = []
            for item in content:
                item_type = item["type"]
                if item_type == "text":
                    text_parts.append(item["text"])
                elif item_type == "video":
                    videos.append(item["video"])
                    video_settings.append({key: item[key] for key in ("fps", "max_frames", "max_pixels") if key in item})
                elif item_type == "image":
                    images.append(item["image"])
                elif item_type == "audio":
                    audios.append(item["audio"])
                else:
                    raise ValueError(f"Unsupported media type: {item_type}")
            text = "".join(text_parts)
        messages.append({"role": message["role"], "content": text})
    return {"messages": messages, "videos": videos, "images": images, "audios": audios}


def append_interaction(
    interactions: list[dict[str, Any]],
    stage: str,
    round_index: int | None,
    request_conversation: list[dict[str, Any]],
    parsed: dict[str, Any] | None,
    raw: str | None,
    *,
    attempt_index: int | None = None,
    status: str = "ok",
    error: Exception | None = None,
    duration_s: float | None = None,
) -> None:
    response: dict[str, Any] = {"parsed": parsed, "raw": raw}
    if error is not None:
        response["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    record = {
        "stage": stage,
        "round_index": round_index,
        "request": serialize_request_conversation(request_conversation),
        "response": response,
        "status": status,
    }
    if attempt_index is not None:
        record["attempt_index"] = attempt_index
    if duration_s is not None:
        record["duration_s"] = duration_s
    interactions.append(record)


def dump_model_interactions(
    output_dir: Path,
    clip_index: int,
    clip: str,
    interactions: list[dict[str, Any]],
) -> None:
    dump_json(
        output_dir / "02_model_interactions.json",
        {
            "clip_index": clip_index,
            "video_path": clip,
            "interactions": interactions,
            "final_request_context": interactions[-1]["request"] if interactions else None,
        },
    )


def image_placeholders(count: int, offset: int = 0) -> str:
    """Label attached images in conversation order, including the trailing newline."""
    return "".join(f"image_{offset + index + 1}: <image>\n" for index in range(count))


def review_image_mapping(items: list[dict[str, Any]], offset: int, *, frames: bool = False) -> str:
    return "\n".join(
        f"image_{offset + index + 1}: detection_index={item['detection_index']}, "
        + (f"subject_id={item['subject_id']}, timestamp={item['timestamp_sec']:.1f}"
           if frames else f"proposed_id={item['subject_id']}")
        for index, item in enumerate(items)
    )


def timestamp_request(
    repository: dict[str, Any], clip: str, clip_index: int, clip_count: int
) -> tuple[str, list[str], list[str]]:
    mapping, references = gallery.repository_media(repository)
    reference_label_text = (
        "Repository reference images are labeled from image_1."
        if references
        else "No repository reference images are attached."
    )
    prompt = (
        f"<video>\n"
        + image_placeholders(len(references))
        + f"Process clip {clip_index} of {clip_count}.\n"
        f"The attached video is the current clip. {reference_label_text}\n\n"
        "Current repository:\n"
        f"{json.dumps(gallery.repository_prompt_view(repository), ensure_ascii=False, indent=2)}\n\n"
        f"Reference images:\n{mapping}\n\n"
        "Return JSON matching the supplied schema."
    )
    return prompt, [clip], references


def frame_review_request(
    repository: dict[str, Any],
    repository_patch: dict[str, Any],
    frames: list[dict[str, Any]],
    clip: str,
    clip_index: int,
    clip_count: int,
    image_offset: int,
) -> tuple[str, list[str], list[str]]:
    mapping = review_image_mapping(frames, image_offset, frames=True)
    prompt = (
        image_placeholders(len(frames), image_offset)
        + f"Review full frames for clip {clip_index} of {clip_count}.\n\n"
        "The video from the first turn is the current clip. Use it only when a "
        "full frame is unsuitable and needs a replacement timestamp.\n\n"
        "Current repository:\n"
        f"{json.dumps(gallery.repository_prompt_view(repository), ensure_ascii=False, indent=2)}\n\n"
        "Timestamp-stage repository patch:\n"
        f"{json.dumps(gallery.model_prompt_patch_view(repository_patch), ensure_ascii=False, indent=2)}\n\n"
        f"Frame images:\n{mapping}\n\n"
        "Judge only whether each full frame contains the intended identity and "
        "is suitable for bbox annotation. If it is suitable, return a normalized "
        "bbox for that same detection_index immediately. If it is not suitable, "
        "return only a corrected current-clip timestamp for that same detection_index."
    )
    return prompt, [], [frame["frame_path"] for frame in frames]


def crop_review_request(
    repository: dict[str, Any],
    repository_patch: dict[str, Any],
    crops: list[dict[str, Any]],
    clip: str,
    clip_index: int,
    clip_count: int,
    image_offset: int,
) -> tuple[str, list[str], list[str]]:
    crop_mapping = review_image_mapping(crops, image_offset)
    ref_mapping, references = gallery.repository_media(
        repository,
        {str(crop["subject_id"]) for crop in crops},
        image_offset + len(crops),
    )
    prompt = (
        image_placeholders(len(crops) + len(references), image_offset)
        + f"Review detections for clip {clip_index} of {clip_count}.\n"
        "The video from the first turn is the current clip. Crop images come "
        "first, followed by relevant existing repository reference images when "
        "the crop updates an existing identity.\n\n"
        "Current repository:\n"
        f"{json.dumps(gallery.repository_prompt_view(repository), ensure_ascii=False, indent=2)}\n\n"
        "Repository patch:\n"
        f"{json.dumps(gallery.model_prompt_patch_view(repository_patch), ensure_ascii=False, indent=2)}\n\n"
        f"Crop images:\n{crop_mapping or 'No crops.'}\n\n"
        f"Relevant repository reference mapping:\n{ref_mapping}\n\n"
        "Verify crop clarity and identity correctness, then make the final "
        "update_reference decision. Return exactly one review for every "
        "detection_index. For rejected crops, inspect the current clip and "
        "return corrected timestamp_s and bounding_box values for the same "
        "detection_index. Because the source full frame was already accepted, "
        "keep timestamp_s unchanged whenever a better bounding_box on the same "
        "frame can fix the crop; change timestamp_s only when that frame cannot "
        "yield a correct discriminative crop. Do not reconsider or modify "
        "repository membership, promotion, events, IDs, or descriptions."
    )
    return prompt, [], [*[crop["crop_path"] for crop in crops], *references]


def mixed_review_request(
    repository: dict[str, Any],
    repository_patch: dict[str, Any],
    crops: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    clip: str,
    clip_index: int,
    clip_count: int,
    image_offset: int,
) -> tuple[str, list[str], list[str]]:
    crop_mapping = review_image_mapping(crops, image_offset)
    frame_mapping = review_image_mapping(frames, image_offset + len(crops), frames=True)
    ref_mapping, references = gallery.repository_media(
        repository,
        {str(crop["subject_id"]) for crop in crops},
        image_offset + len(crops) + len(frames),
    )
    prompt = (
        image_placeholders(len(crops) + len(frames) + len(references), image_offset)
        + f"Review crop images and corrected full frames for clip "
        f"{clip_index} of {clip_count}.\n"
        "The video from the first turn is the current clip. Crop images come "
        "first, followed by corrected full-frame images, then existing "
        "repository reference images only for crop identities that already "
        "have references.\n\n"
        "Current repository:\n"
        f"{json.dumps(gallery.repository_prompt_view(repository), ensure_ascii=False, indent=2)}\n\n"
        "Repository patch:\n"
        f"{json.dumps(gallery.model_prompt_patch_view(repository_patch), ensure_ascii=False, indent=2)}\n\n"
        f"Crop images needing crop review:\n{crop_mapping or 'No crops.'}\n\n"
        "Corrected full-frame images needing bbox-or-timestamp review:\n"
        f"{frame_mapping or 'No corrected frames.'}\n\n"
        f"Relevant repository reference mapping:\n{ref_mapping}\n\n"
        "For crop images, verify crop clarity and identity correctness, then "
        "make the final update_reference decision in crop_reviews. Because "
        "each crop source full frame was already accepted, keep timestamp_s "
        "unchanged for rejected crops whenever a better bounding_box on the "
        "same frame can fix the crop; change timestamp_s only when that frame "
        "cannot yield a correct discriminative crop. For corrected full-frame "
        "images, return a bbox immediately if the timestamp is suitable, or "
        "return another timestamp_s if it is not. Do not reconsider or modify "
        "repository membership, promotion, events, IDs, or descriptions."
    )
    return (
        prompt,
        [],
        [
            *[crop["crop_path"] for crop in crops],
            *[frame["frame_path"] for frame in frames],
            *references,
        ],
    )


def apply_accepts(
    repository: dict[str, Any],
    repository_patch: dict[str, Any],
    reviews: dict[int, dict[str, Any]],
    crops: list[dict[str, Any]],
    clip_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    accepted_indices = {index for index, review in reviews.items() if review["accepted"]}
    if not accepted_indices:
        return repository, repository_patch
    patch = gallery.filter_repository_patch_by_indices(repository_patch, accepted_indices)
    accepted_crops = [
        crop for crop in crops if int(crop["detection_index"]) in accepted_indices
    ]
    review_payload = {"reviews": [reviews[index] for index in sorted(accepted_indices)]}
    repository = gallery.apply_review_patch(
        repository,
        patch,
        gallery.review_patch_payload(review_payload),
        accepted_crops,
        clip_index,
    )
    return repository, gallery.remove_repository_patch_indices(repository_patch, accepted_indices)


def repository_subject_ids(repository: Any) -> set[str]:
    if not isinstance(repository, dict):
        return set()
    subjects = repository.get("subjects", {})
    if not isinstance(subjects, dict):
        return set()
    return {str(subject_id) for subject_id in subjects if str(subject_id)}


def repository_subject_count(repository: Any) -> int:
    return len(repository_subject_ids(repository))


def repository_candidate_count(repository: Any) -> int:
    if not isinstance(repository, dict):
        return 0
    candidates = repository.get("candidates", {})
    if not isinstance(candidates, dict):
        return 0
    return len([subject_id for subject_id in candidates if str(subject_id)])


def clip_result_record(
    clip_index: int,
    clip: str,
    repository_patch: dict[str, Any],
    detections: list[dict[str, Any]],
    reviews: dict[int, dict[str, Any]],
    bbox_revision_count: int,
    repository: dict[str, Any],
    *,
    status: str = "completed",
    error: Exception | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "clip_index": clip_index,
        "video_path": clip,
        "repository_patch": repository_patch,
        "detections": detections,
        "review": {"reviews": list(reviews.values())},
        "bbox_review_timestamp_revision_count": bbox_revision_count,
        "repository_after": repository,
    }
    if timing is not None:
        result["timing"] = timing
    if status != "completed":
        result["status"] = status
    if error is not None:
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    return result


def finalize_clip_timing(
    timing: dict[str, Any],
    interactions: list[dict[str, Any]],
    clip_start: float,
) -> dict[str, Any]:
    result = deepcopy(timing)
    total_s = elapsed_seconds(clip_start)
    model_s = round(
        sum(
            float(item.get("duration_s", 0.0) or 0.0)
            for item in interactions
        ),
        3,
    )
    result.update(
        {
            "total_s": total_s,
            "model_interaction_s": model_s,
            "model_interaction_count": len(interactions),
            "failed_validation_count": sum(
                1 for item in interactions if item.get("status") != "ok"
            ),
            "local_orchestration_s": round(max(total_s - model_s, 0.0), 3),
        }
    )
    return result


def infer_clip(
    model: Model,
    repository: dict[str, Any],
    clip: str,
    clip_index: int,
    clip_count: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    conversation: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    interactions: list[dict[str, Any]] = []
    repository_patch: dict[str, Any] = {
        "subjects_upsert": [],
        "candidates_upsert": [],
        "candidate_deletes": [],
    }
    pending_frame_indices: list[int] = []
    pending_crop_indices: list[int] = []
    all_crops: list[dict[str, Any]] = []
    all_reviews: dict[int, dict[str, Any]] = {}
    bbox_revision_count = 0
    clip_start = time.perf_counter()
    timing: dict[str, Any] = {
        "probe_s": 0.0,
        "rounds": [],
    }

    try:
        probe_start = time.perf_counter()
        fps, _, frame_count = media.probe_video(Path(clip))
        timing["probe_s"] = elapsed_seconds(probe_start)
        max_timestamp = max((frame_count - 1) / fps, 0.0)

        prompt, videos, images = timestamp_request(repository, clip, clip_index, clip_count)
        timestamp_message = make_message(
            "user",
            prompt,
            videos=videos,
            images=images,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        payload, raw, request_conversation = generate_turn(
            model,
            conversation,
            timestamp_message,
            lambda value: gallery.validate_timestamp_proposal_response(value, max_timestamp),
            args.validation_retries,
            stage="timestamp_proposal",
            round_index=None,
            interactions=interactions,
            interaction_output_dir=output_dir,
            clip_index=clip_index,
            clip=clip,
        )
        repository_patch = gallery.validate_timestamp_proposal_response(payload, max_timestamp)
        dump_json(output_dir / "01_timestamp_proposal.json", {"parsed": payload, "raw": raw})

        repository, repository_patch = gallery.apply_immediate_promotions(
            repository, repository_patch, clip_index
        )
        pending_frame_indices = gallery.visual_review_indices(repository, repository_patch)

        for round_index in range(args.max_review_rounds + 1):
            if not pending_frame_indices and not pending_crop_indices:
                break
            round_timing: dict[str, Any] = {
                "round_index": round_index,
                "pending_frame_count_start": len(pending_frame_indices),
                "pending_crop_count_start": len(pending_crop_indices),
                "frame_extract_s": 0.0,
                "crop_extract_s": 0.0,
                "retry_frame_extract_s": 0.0,
            }
            round_start = time.perf_counter()
            round_dir = output_dir / "03_rounds" / f"{round_index:03d}_round"
            detections = {
                int(item["detection_index"]): item
                for item in gallery.flatten_detections(repository_patch)
            }

            if pending_frame_indices and not pending_crop_indices:
                frame_extract_start = time.perf_counter()
                frames = [
                    media.extract_frame(Path(clip), detections[index], index, round_dir)
                    for index in pending_frame_indices
                ]
                round_timing["frame_extract_s"] = elapsed_seconds(frame_extract_start)
                dump_json(round_dir / "01_frames.json", frames)
                prompt, videos, images = frame_review_request(
                    repository,
                    repository_patch,
                    frames,
                    clip,
                    clip_index,
                    clip_count,
                    count_conversation_images(conversation),
                )
                frame_message = make_message(
                    "user",
                    prompt,
                    videos=videos,
                    images=images,
                    fps=args.fps,
                    max_frames=args.max_frames,
                )
                requested = set(pending_frame_indices)
                frame_payload, frame_raw, request_conversation = generate_turn(
                    model,
                    conversation,
                    frame_message,
                    lambda value: gallery.validate_frame_review_bbox_response(
                        value, requested, max_timestamp, detections
                    ),
                    args.validation_retries,
                    stage="frame_review",
                    round_index=round_index,
                    interactions=interactions,
                    interaction_output_dir=output_dir,
                    clip_index=clip_index,
                    clip=clip,
                )
                boxes, timestamp_revisions = gallery.validate_frame_review_bbox_response(
                    frame_payload, requested, max_timestamp, detections
                )
                dump_json(
                    round_dir / "02_frame_review.json",
                    {"parsed": frame_payload, "raw": frame_raw},
                )
                repository_patch = gallery.apply_bbox_response(repository_patch, boxes)
                repository_patch = gallery.apply_timestamp_revisions(
                    repository_patch, timestamp_revisions
                )
                pending_crop_indices = sorted({*pending_crop_indices, *boxes})
                pending_frame_indices = sorted(timestamp_revisions)

            if not pending_crop_indices:
                round_timing["pending_frame_count_end"] = len(pending_frame_indices)
                round_timing["pending_crop_count_end"] = len(pending_crop_indices)
                round_timing["rejected_review_count"] = 0
                round_timing["duration_s"] = elapsed_seconds(round_start)
                timing["rounds"].append(round_timing)
                if pending_frame_indices and round_index >= args.max_review_rounds:
                    raise RuntimeError(
                        "Frame review exceeded max rounds for indices: "
                        f"{pending_frame_indices}"
                    )
                continue

            detections = {
                int(item["detection_index"]): item
                for item in gallery.flatten_detections(repository_patch)
            }
            crop_extract_start = time.perf_counter()
            crops = [
                media.extract_and_crop(Path(clip), detections[index], index, round_dir)
                for index in pending_crop_indices
            ]
            round_timing["crop_extract_s"] = elapsed_seconds(crop_extract_start)
            all_crops.extend(crops)
            dump_json(round_dir / "04_crops.json", crops)

            retry_frame_extract_start = time.perf_counter()
            retry_frames = [
                media.extract_frame(Path(clip), detections[index], index, round_dir)
                for index in pending_frame_indices
            ]
            round_timing["retry_frame_extract_s"] = elapsed_seconds(
                retry_frame_extract_start
            )
            if retry_frames:
                dump_json(round_dir / "03_retry_frames.json", retry_frames)

            if retry_frames:
                prompt, videos, images = mixed_review_request(
                    repository,
                    repository_patch,
                    crops,
                    retry_frames,
                    clip,
                    clip_index,
                    clip_count,
                    count_conversation_images(conversation),
                )
                mixed_message = make_message(
                    "user",
                    prompt,
                    videos=videos,
                    images=images,
                    fps=args.fps,
                    max_frames=args.max_frames,
                )
                crop_requested = set(pending_crop_indices)
                frame_requested = set(pending_frame_indices)
                mixed_payload, mixed_raw, request_conversation = generate_turn(
                    model,
                    conversation,
                    mixed_message,
                    lambda value: gallery.validate_mixed_crop_frame_review_response(
                        value,
                        crop_requested,
                        frame_requested,
                        max_timestamp,
                        detections,
                    ),
                    args.validation_retries,
                    stage="mixed_review",
                    round_index=round_index,
                    interactions=interactions,
                    interaction_output_dir=output_dir,
                    clip_index=clip_index,
                    clip=clip,
                )
                reviews, boxes, timestamp_revisions = (
                    gallery.validate_mixed_crop_frame_review_response(
                        mixed_payload,
                        crop_requested,
                        frame_requested,
                        max_timestamp,
                        detections,
                    )
                )
                dump_json(
                    round_dir / "05_mixed_review.json",
                    {"parsed": mixed_payload, "raw": mixed_raw},
                )
                repository_patch = gallery.apply_bbox_response(repository_patch, boxes)
                repository_patch = gallery.apply_timestamp_revisions(
                    repository_patch, timestamp_revisions
                )
                frame_box_indices = set(boxes)
                pending_frame_indices = sorted(timestamp_revisions)
            else:
                prompt, videos, images = crop_review_request(
                    repository,
                    repository_patch,
                    crops,
                    clip,
                    clip_index,
                    clip_count,
                    count_conversation_images(conversation),
                )
                crop_message = make_message(
                    "user",
                    prompt,
                    videos=videos,
                    images=images,
                    fps=args.fps,
                    max_frames=args.max_frames,
                )
                crop_requested = set(pending_crop_indices)
                crop_payload, crop_raw, request_conversation = generate_turn(
                    model,
                    conversation,
                    crop_message,
                    lambda value: gallery.validate_crop_review_response(
                        value, crop_requested, max_timestamp, detections
                    ),
                    args.validation_retries,
                    stage="crop_review",
                    round_index=round_index,
                    interactions=interactions,
                    interaction_output_dir=output_dir,
                    clip_index=clip_index,
                    clip=clip,
                )
                reviews = gallery.validate_crop_review_response(
                    crop_payload, crop_requested, max_timestamp, detections
                )
                dump_json(
                    round_dir / "05_crop_review.json",
                    {"parsed": crop_payload, "raw": crop_raw},
                )
                frame_box_indices = set()

            all_reviews.update(reviews)
            repository, repository_patch = apply_accepts(
                repository, repository_patch, reviews, crops, clip_index
            )
            rejected_reviews = [
                review for review in reviews.values() if not review["accepted"]
            ]
            if rejected_reviews:
                bbox_revision_count += gallery.count_crop_review_timestamp_revisions(
                    repository_patch, rejected_reviews
                )
                repository_patch = gallery.apply_crop_review_revisions(
                    repository_patch, rejected_reviews
                )
            pending_crop_indices = sorted(
                {
                    *frame_box_indices,
                    *(int(review["detection_index"]) for review in rejected_reviews),
                }
            )
            round_timing["pending_frame_count_end"] = len(pending_frame_indices)
            round_timing["pending_crop_count_end"] = len(pending_crop_indices)
            round_timing["rejected_review_count"] = len(rejected_reviews)
            round_timing["duration_s"] = elapsed_seconds(round_start)
            timing["rounds"].append(round_timing)
            if (
                pending_frame_indices or pending_crop_indices
            ) and round_index >= args.max_review_rounds:
                raise RuntimeError(
                    "Visual review exceeded max rounds for indices: "
                    f"{sorted({*pending_frame_indices, *pending_crop_indices})}"
                )

        if pending_frame_indices or pending_crop_indices:
            raise RuntimeError(
                "Unreviewed detections remain: "
                f"{sorted({*pending_frame_indices, *pending_crop_indices})}"
            )

        if gallery.flatten_detections(repository_patch):
            repository = gallery.apply_review_patch(
                repository,
                repository_patch,
                gallery.review_patch_payload({"reviews": []}),
                [],
                clip_index,
            )

        result = clip_result_record(
            clip_index,
            clip,
            repository_patch,
            all_crops,
            all_reviews,
            bbox_revision_count,
            repository,
            timing=finalize_clip_timing(timing, interactions, clip_start),
        )
        dump_json(output_dir / "01_result.json", result)
        dump_model_interactions(output_dir, clip_index, clip, interactions)
        return repository, result

    except Exception as exc:
        result = clip_result_record(
            clip_index,
            clip,
            repository_patch,
            all_crops,
            all_reviews,
            bbox_revision_count,
            repository,
            status="failed",
            error=exc,
            timing=finalize_clip_timing(timing, interactions, clip_start),
        )
        dump_json(output_dir / "01_result.json", result)
        dump_model_interactions(output_dir, clip_index, clip, interactions)
        raise PartialRepositoryBuildError(
            f"Subject repository build failed at clip {clip_index}: {exc}",
            repository,
            result,
            exc,
        ) from exc


def infer_repository(
    model: Model,
    clips: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    repository, _metadata = infer_repository_with_metadata(model, clips, output_dir, args)
    return repository


def infer_repository_with_metadata(
    model: Model,
    clips: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository: dict[str, Any] = {"subjects": {}, "candidates": {}}
    clip_results: list[dict[str, Any]] = []
    repository_errors: list[dict[str, Any]] = []
    for offset, clip in enumerate(clips):
        clip_index = offset + 1
        clip_dir = output_dir / "subject_repository" / f"{clip_index:04d}_clip"
        try:
            repository, result = infer_clip(
                model, repository, clip, clip_index, len(clips), clip_dir, args
            )
        except PartialRepositoryBuildError as exc:
            if is_backend_unhealthy_error(exc.cause):
                raise exc.cause from exc
            repository = deepcopy(exc.repository)
            result = deepcopy(exc.clip_result)
            clip_results.append(result)
            dump_json(output_dir / "subject_repository.json", repository)
            repository_errors.append(
                {
                    "clip_index": clip_index,
                    "video_path": clip,
                    "error": str(exc.cause),
                    "error_type": type(exc.cause).__name__,
                }
            )
            continue
        clip_results.append(result)
        dump_json(output_dir / "subject_repository.json", repository)
    dump_json(output_dir / "subject_repository" / "clip_results.json", clip_results)
    metadata: dict[str, Any] = {"repository_status": "completed"}
    if repository_errors:
        first_error = repository_errors[0]
        metadata = {
            "repository_status": "partial_failed",
            "repository_failed_clip_indices": [
                item["clip_index"] for item in repository_errors
            ],
            "repository_errors": repository_errors,
            "repository_failed_clip_index": first_error["clip_index"],
            "repository_error": first_error["error"],
            "repository_error_type": first_error["error_type"],
        }
    return repository, metadata


def write_caption_outputs(output_dir: Path, captions: list[dict[str, Any]]) -> None:
    dump_json(output_dir / "captions.json", captions)
    text = "\n\n".join(item["caption"] for item in captions)
    (output_dir / "full_caption.txt").write_text(
        f"{text}\n" if text else "", encoding="utf-8"
    )


def caption_prompt(
    clip_index: int,
    clip_count: int,
    previous_caption: str,
    previous_summary: dict[str, Any],
) -> str:
    return (
        f"Caption clip {clip_index} of {clip_count}.\n"
        "Cumulative structured summary through the previous clip:\n"
        f"{json.dumps(previous_summary, ensure_ascii=False, indent=2)}\n\n"
        "Previous clip caption:\n"
        f"{previous_caption or 'No previous clip; this is the beginning of the video.'}\n\n"
        "The final long-video caption is formed by directly concatenating all "
        "clip captions in order, with no rewriting pass. Generate the current "
        "caption as the next coherent passage after the previous caption. Do not "
        "repeat content already covered unless a brief reference is necessary "
        "for continuity; focus on new actions, changes, and information. Include "
        "meaningful background audio, camera and shot changes, and relevant people, "
        "animals, or objects without confirmed IDs. Cover brief actions, background "
        "reactions, transitions, and meaningful opening and final shots rather than "
        "only the dominant storyline. Keep each actor, recipient, speaker, reaction, "
        "and handled object correctly assigned. Describe confirmed subjects' actions "
        "at fine granularity. A reference image identifies a subject but does not "
        "replace appearance description: state the subject's visible appearance at "
        "their first appearance and whenever it changes. Speech may be summarized or "
        "transcribed verbatim; any verbatim transcription must preserve the language "
        "actually spoken. Use conservative wording for uncertain colors, clothing, "
        "objects, speakers, and causal relations. Silently check coverage and "
        "accuracy before returning the JSON. Only captions appear in the final "
        "output, so every new background and established fact must first be expressed "
        "in the caption before it is added to the structured summary. Use the "
        "cumulative summary as global memory of important information already "
        "covered, and use the previous caption for immediate continuity. Do not "
        "repeat an established fact unless the current clip changes, corrects, or "
        "meaningfully develops it. Then update the cumulative structured summary "
        "through the end of this clip."
    )


def validate_caption(
    payload: dict[str, Any], confirmed_ids: set[str], candidate_ids: set[str]
) -> dict[str, Any]:
    if set(payload) != {"caption", "structured_summary"}:
        raise ValueError("Caption response has unexpected top-level keys")
    if not isinstance(payload["caption"], str) or not payload["caption"].strip():
        raise ValueError("caption must be non-empty")
    summary = payload["structured_summary"]
    if not isinstance(summary, dict) or set(summary) != set(EMPTY_SUMMARY):
        raise ValueError("Invalid structured_summary")
    if not all(isinstance(value, list) for value in summary.values()):
        raise ValueError("Summary fields must be lists")
    payload_text = json.dumps(payload, ensure_ascii=False)
    legacy_ids = set(LEGACY_SUBJECT_ID_SEARCH_RE.findall(payload_text))
    if legacy_ids:
        raise ValueError(f"Legacy angle-bracket IDs used in caption: {sorted(legacy_ids)}")
    used_ids = set(SUBJECT_ID_SEARCH_RE.findall(payload_text))
    if used_ids & candidate_ids:
        raise ValueError(f"Candidate IDs used in caption: {sorted(used_ids & candidate_ids)}")
    if used_ids - confirmed_ids:
        raise ValueError(f"Invented IDs used in caption: {sorted(used_ids - confirmed_ids)}")
    return payload


def infer_captions(
    model: Model,
    clips: list[str],
    repository: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    references = []
    reference_ids = []
    for subject_id, entry in sorted(repository["subjects"].items()):
        path = entry.get("reference_image_path")
        if path and Path(path).is_file():
            reference_ids.append(subject_id)
            references.append(path)
    labels = "\n".join(f"{subject_id}: <image>" for subject_id in reference_ids)
    captions: list[dict[str, Any]] = []
    previous_caption = ""
    previous_summary = deepcopy(EMPTY_SUMMARY)
    for clip_offset, clip in enumerate(clips):
        caption_start = time.perf_counter()
        caption_timing: dict[str, Any] = {
            "model_generate_s": 0.0,
            "validate_s": 0.0,
        }
        clip_index = clip_offset + 1
        prompt = caption_prompt(
            clip_index, len(clips), previous_caption, previous_summary
        )
        user_text = f"{labels + chr(10) if labels else ''}{prompt}"
        conversation = [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            make_message(
                "user",
                user_text,
                videos=[clip],
                images=references,
                fps=args.fps,
                max_frames=args.max_frames,
            ),
        ]
        raw: str | None = None
        model_start = time.perf_counter()
        try:
            payload, raw = model.generate_caption(conversation)
            caption_timing["model_generate_s"] = elapsed_seconds(model_start)
            validate_start = time.perf_counter()
            result = validate_caption(
                payload, set(repository["subjects"]), set(repository["candidates"])
            )
            caption_timing["validate_s"] = elapsed_seconds(validate_start)
        except Exception as exc:
            raw = getattr(exc, "raw_response", raw)
            if caption_timing["model_generate_s"] == 0.0:
                caption_timing["model_generate_s"] = elapsed_seconds(model_start)
            caption_timing["total_s"] = elapsed_seconds(caption_start)
            dump_json(
                output_dir / "captions" / f"{clip_index:04d}_clip.error.json",
                {
                    "clip_index": clip_index,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "raw": raw,
                    "timing": caption_timing,
                },
            )
            write_caption_outputs(output_dir, captions)
            raise
        caption_timing["total_s"] = elapsed_seconds(caption_start)
        record = {"clip_index": clip_index, "timing": caption_timing, **result}
        captions.append(record)
        dump_json(
            output_dir / "captions" / f"{clip_index:04d}_clip.json",
            {"parsed": result, "raw": raw, "timing": caption_timing},
        )
        write_caption_outputs(output_dir, captions)
        previous_caption = result["caption"]
        previous_summary = result["structured_summary"]
    return captions


def unicode_safe_name(value: str, fallback: str = "video", max_chars: int = 80) -> str:
    name = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af.-]+", "_", value, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("._ ")
    return (name[:max_chars].rstrip("._ ") or fallback)


def infer_dataset_label(video: Path) -> str | None:
    parts = video.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "datasets" and index + 1 < len(parts):
            return unicode_safe_name(parts[index + 1], fallback="dataset", max_chars=40)
    return None


def video_output_name(video: Path, dataset_label: str | None = None) -> str:
    stem = unicode_safe_name(video.stem)
    dataset = unicode_safe_name(dataset_label, fallback="dataset", max_chars=40) if dataset_label else None
    digest = hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{dataset + '_' if dataset else ''}{stem}_{digest}"


def video_output_dir(video: Path, args: argparse.Namespace) -> Path:
    video = video.resolve()
    dataset_label = args.dataset_label or infer_dataset_label(video)
    return args.output_dir / video_output_name(video, dataset_label)


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def completed_summary(output_dir: Path) -> dict[str, Any] | None:
    summary = read_json_if_exists(output_dir / "summary.json")
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        return None
    expected = (
        "subject_repository.json",
        "repository.json",
        "captions.json",
        "full_caption.txt",
    )
    if not all((output_dir / name).exists() for name in expected):
        return None
    return summary


def failure_record(video: Path, args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    record = {
        "video": str(video.resolve()),
        "status": "failed",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    try:
        record["output_dir"] = str(video_output_dir(video, args).resolve())
    except Exception:
        pass
    output_dir = record.get("output_dir")
    if isinstance(output_dir, str):
        try:
            dump_json(Path(output_dir) / "summary.json", record)
        except Exception:
            pass
    return record


def infer_video(video: Path, args: argparse.Namespace) -> dict[str, Any]:
    video_start = time.perf_counter()
    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    dataset_label = args.dataset_label or infer_dataset_label(video)
    output_dir = video_output_dir(video, args)
    clip_base_dir = args.clip_dir / video_output_name(video, dataset_label)
    clip_dir = clip_base_dir / video.stem
    if output_dir.exists() and not args.overwrite:
        summary = completed_summary(output_dir)
        if summary is not None:
            print(f"[video:skip-completed] {video}", flush=True)
            return summary
        print(f"[video:rerun-incomplete] {video}", flush=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        if not args.preserve_clips_on_rerun:
            shutil.rmtree(clip_dir, ignore_errors=True)
    elif args.overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
        if not args.preserve_clips_on_rerun:
            shutil.rmtree(clip_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[video:start] {video}", flush=True)
    split_start = time.perf_counter()
    _segment_len, _n_segments, split_out_dir = split_video_with_retry(
        video,
        clip_base_dir,
        target_seconds=args.clip_seconds,
        max_len=args.max_clip_seconds,
    )
    split_s = elapsed_seconds(split_start)
    clips = [
        str(path.resolve())
        for path in sorted(Path(split_out_dir).glob("*.mp4"))
    ]
    if not clips:
        raise RuntimeError(f"No split clips found in {split_out_dir}")

    model_init_start = time.perf_counter()
    model = Model(args)
    model_init_s = elapsed_seconds(model_init_start)
    dump_json(
        output_dir / "run.json",
        {
            "video": str(video),
            "dataset": dataset_label,
            "clips": clips,
            "model": model.client.models[0] if model.client.models else None,
        },
    )
    repository_start = time.perf_counter()
    repository, repository_metadata = infer_repository_with_metadata(
        model, clips, output_dir, args
    )
    repository_s = elapsed_seconds(repository_start)
    dump_json(output_dir / "repository.json", repository)
    captions_start = time.perf_counter()
    captions = infer_captions(model, clips, repository, output_dir, args)
    captions_s = elapsed_seconds(captions_start)
    total_s = elapsed_seconds(video_start)
    timing = {
        "total_s": total_s,
        "split_s": split_s,
        "model_init_s": model_init_s,
        "repository_s": repository_s,
        "captions_s": captions_s,
    }
    summary = {
        "video": str(video),
        "dataset": dataset_label,
        "output_dir": str(output_dir.resolve()),
        "clips": len(clips),
        "subjects": repository_subject_count(repository),
        "candidates": repository_candidate_count(repository),
        "captions": len(captions),
        "status": "completed",
        "timing": timing,
        **repository_metadata,
    }
    dump_json(output_dir / "summary.json", summary)
    print(
        f"[video:done] {video} "
        f"total_s={total_s:.3f} split_s={split_s:.3f} "
        f"repository_s={repository_s:.3f} captions_s={captions_s:.3f}",
        flush=True,
    )
    return summary


def main() -> int:
    args = parse_args()
    args.video = list(dict.fromkeys(path.resolve() for path in args.video))
    args.clip_dir = args.clip_dir or args.output_dir / "clips"
    for name in ("clip_seconds", "max_clip_seconds", "max_tokens", "fps", "max_frames", "validation_retries"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace(chr(95), chr(45))} must be positive")
    if args.max_clip_seconds < args.clip_seconds:
        raise ValueError("--max-clip-seconds must be >= --clip-seconds")
    if args.max_review_rounds < 0:
        raise ValueError("--max-review-rounds must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    # Resolve Swift's lazy exports on the main thread before workers import them.
    # Keep this after argument parsing so --help does not load inference dependencies.
    from swift import InferClient, InferRequest, InferStats, RequestConfig  # noqa: F401

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.clip_dir.mkdir(parents=True, exist_ok=True)

    worker_count = min(args.workers, len(args.video))
    summaries: list[dict[str, Any] | None] = [None] * len(args.video)
    failures: list[dict[str, Any]] = []
    backend_unhealthy_detected = False

    if worker_count == 1:
        for index, video in enumerate(args.video):
            try:
                summaries[index] = infer_video(video, args)
            except Exception as exc:
                failures.append(failure_record(video, args, exc))
                if args.fail_fast_on_backend_error and is_backend_unhealthy_error(exc):
                    backend_unhealthy_detected = True
                    log_backend_unhealthy_event(
                        args,
                        video=video,
                        exc=exc,
                        worker_count=worker_count,
                        completed_count=sum(summary is not None for summary in summaries),
                        failure_count=len(failures),
                    )
                    print(
                        f"[backend:unhealthy] {video}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                break
    else:
        print(
            f"[parallel] running {len(args.video)} videos with {worker_count} workers",
            flush=True,
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
        future_to_index = {
            executor.submit(infer_video, video, args): index
            for index, video in enumerate(args.video)
        }
        wait_for_running = True
        try:
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                video = args.video[index]
                try:
                    summaries[index] = future.result()
                except Exception as exc:
                    failures.append(failure_record(video, args, exc))
                    print(f"[video:failed] {video}: {exc}", file=sys.stderr, flush=True)
                    if args.fail_fast_on_backend_error and is_backend_unhealthy_error(exc):
                        backend_unhealthy_detected = True
                        log_backend_unhealthy_event(
                            args,
                            video=video,
                            exc=exc,
                            worker_count=worker_count,
                            completed_count=sum(summary is not None for summary in summaries),
                            failure_count=len(failures),
                        )
                        print(
                            f"[backend:unhealthy] {video}: {exc}; cancelling pending videos",
                            file=sys.stderr,
                            flush=True,
                        )
                        for pending in future_to_index:
                            if pending is not future:
                                pending.cancel()
                        wait_for_running = False
                        break
        finally:
            executor.shutdown(wait=wait_for_running, cancel_futures=True)

    final_summaries = [summary for summary in summaries if summary is not None]
    final_summaries.extend(failures)
    dump_json(args.output_dir / "summary.json", final_summaries)
    if backend_unhealthy_detected:
        print(
            "ERROR: backend unhealthy; exiting with status 99 for supervisor restart",
            file=sys.stderr,
            flush=True,
        )
        os._exit(99)
    if failures:
        failed = ", ".join(item["video"] for item in failures)
        raise RuntimeError(f"{len(failures)} video(s) failed: {failed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
