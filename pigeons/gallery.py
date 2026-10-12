"""Identity-gallery prompts, response validation, state updates, and promotion."""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import media


SUBJECT_ID_RE = re.compile(r"^[A-Z]+_[A-Z]$")

SUBJECT_ID_LIKE_SEARCH_RE = re.compile(
    r"\b[A-Z][A-Z0-9_]*_[A-Z0-9_]+\b"
)

LEGACY_SUBJECT_ID_SEARCH_RE = re.compile(
    r"<[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+>"
)

REPOSITORY_TEXT_FIELDS = {
    "subject_description",
    "event",
    "promotion_reason",
}

EXTERNAL_TIMESTAMP_FIELD = "timestamp_s"

GALLERY_SYSTEM_PROMPT = (
    "Maintain the confirmed-subject and candidate repositories from the video "
    "and repository context. Review crop tool results when provided. When the "
    "task asks for natural-language analysis before JSON, keep that analysis "
    "concise and directly tied to the final JSON decisions, including relevant "
    "subject IDs, evidence, and timestamps. Otherwise return the requested JSON "
    "only. Subject IDs use the TYPE_LETTER namespace without a prefix, angle "
    "brackets, or digits."
)


# Prompt presentation and reference images


def without_artifact_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_artifact_paths(item)
            for key, item in value.items()
            if key not in {
                "crop_path",
                "frame_path",
                "reference_image_path",
                "reference_frame_path",
                "agent_trajectory_path",
            }
        }
    if isinstance(value, list):
        return [without_artifact_paths(item) for item in value]
    return value


def repository_prompt_view(repository: dict[str, Any]) -> dict[str, Any]:
    value = without_artifact_paths(repository)
    for group_name in ("subjects", "candidates"):
        group = value.get(group_name, {})
        if not isinstance(group, dict):
            continue
        for entry in group.values():
            if isinstance(entry, dict):
                entry.pop("reference_frame_timestamp", None)
                entry.pop("bounding_box", None)
                entry.pop("promotion_reason", None)
    return value


def model_prompt_patch_view(repository_patch: dict[str, Any]) -> dict[str, Any]:
    view = without_artifact_paths(repository_patch)
    if not isinstance(view, dict):
        return view
    view = deepcopy(view)
    for field_name in ("subjects_upsert", "candidates_upsert"):
        items = view.get(field_name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item.pop("reference_crop_required", None)
            if "reference_frame_timestamp" in item:
                item[EXTERNAL_TIMESTAMP_FIELD] = item.pop(
                    "reference_frame_timestamp"
                )
            item.pop("promotion_reason", None)
            item.pop("timestamp_sec", None)
            item.pop("bbox_1000", None)
    return view


def repository_media(
    repository: dict[str, Any],
    subject_ids: set[str] | None = None,
    start_index: int = 0,
) -> tuple[str, list[str]]:
    """Build ordered reference images and their matching model-facing labels."""
    lines = []
    paths = []
    for group in ("subjects", "candidates"):
        for subject_id, entry in sorted(repository[group].items()):
            if subject_ids is not None and subject_id not in subject_ids:
                continue
            reference_path = entry.get("reference_image_path")
            if reference_path and Path(reference_path).is_file():
                paths.append(reference_path)
                lines.append(
                    f"image_{start_index + len(paths)}: {group}.{subject_id} reference"
                )
    empty = "No repository reference images." if subject_ids is None else "No relevant repository reference images."
    return "\n".join(lines) or empty, paths


# Subject and timestamp validation


def iter_repository_text_values(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in REPOSITORY_TEXT_FIELDS and isinstance(item, str):
                texts.append(item)
            else:
                texts.extend(iter_repository_text_values(item))
    elif isinstance(value, list):
        for item in value:
            texts.extend(iter_repository_text_values(item))
    return texts


def validate_subject_id_text_references(value: Any, context: str) -> None:
    serialized = json.dumps(
        iter_repository_text_values(value),
        ensure_ascii=False,
    )
    legacy_ids = sorted(set(LEGACY_SUBJECT_ID_SEARCH_RE.findall(serialized)))
    if legacy_ids:
        raise ValueError(
            f"Legacy angle-bracket subject IDs used in {context}: {legacy_ids}"
        )
    malformed_subject_ids = sorted(
        subject_id
        for subject_id in set(SUBJECT_ID_LIKE_SEARCH_RE.findall(serialized))
        if not SUBJECT_ID_RE.fullmatch(subject_id)
    )
    if malformed_subject_ids:
        raise ValueError(
            f"Malformed subject IDs used in {context}: {malformed_subject_ids}"
        )


def validate_repository_subject_ids(repository: Any, context: str) -> None:
    if not isinstance(repository, dict):
        raise ValueError(f"{context} must be an object")
    invalid: list[str] = []
    for group_name in ("subjects", "candidates"):
        group = repository.get(group_name)
        if not isinstance(group, dict):
            raise ValueError(f"{context}.{group_name} must be an object")
        invalid.extend(
            str(subject_id)
            for subject_id in group
            if not SUBJECT_ID_RE.fullmatch(str(subject_id))
        )
    if invalid:
        raise ValueError(
            f"Invalid subject repository IDs in {context}: "
            f"{sorted(invalid)[:20]}"
        )
    overlap = set(repository["subjects"]) & set(repository["candidates"])
    if overlap:
        raise ValueError(
            f"Subject/candidate repository overlap in {context}: "
            f"{sorted(overlap)[:20]}"
        )


def validate_subject_repository(repository: Any, context: str) -> None:
    validate_repository_subject_ids(repository, context)
    validate_subject_id_text_references(repository, context)


def validate_reference_timestamp(
    timestamp: Any,
    subject_id: str,
    max_timestamp: float,
) -> float:
    try:
        normalized = float(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid reference timestamp for {subject_id}: {timestamp!r}"
        ) from exc
    if not math.isfinite(normalized):
        raise ValueError(f"Invalid reference timestamp for {subject_id}: {normalized}")
    if normalized < 0.0 or normalized > max_timestamp:
        raise ValueError(
            f"reference_frame_timestamp for {subject_id} is {normalized:.3f}, "
            f"outside the valid current-clip range [0.000, {max_timestamp:.3f}]"
        )
    return normalized


def normalize_timestamp_01(timestamp: Any, subject_id: str, max_timestamp: float) -> float:
    value = validate_reference_timestamp(timestamp, subject_id, max_timestamp)
    return round(value, 1)


def external_timestamp(item: dict[str, Any], subject_id: str) -> Any:
    if EXTERNAL_TIMESTAMP_FIELD not in item:
        raise ValueError(f"Missing {EXTERNAL_TIMESTAMP_FIELD} for {subject_id}")
    return item[EXTERNAL_TIMESTAMP_FIELD]


# Model response parsing and review validation


def extract_final_json_object(raw: str) -> tuple[dict[str, Any], str]:
    decoder = json.JSONDecoder()
    last_match: tuple[dict[str, Any], int, int] | None = None
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        trailing = raw[index + end:].strip()
        if isinstance(parsed, dict) and trailing in {"", "```", "```json"}:
            last_match = parsed, index, index + end
    if last_match is None:
        raise ValueError("No final JSON object found in timestamp proposal response")
    parsed, json_start, json_end = last_match
    analysis = raw[:json_start].strip()
    analysis = analysis.removesuffix("```json").removesuffix("```").strip()
    parsed["_natural_language_analysis"] = analysis
    return parsed, analysis


def validate_timestamp_proposal_response(
    payload: Any,
    max_timestamp: float,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Timestamp proposal response must be an object")
    if "natural_language_observation" in payload:
        raise ValueError(
            "natural_language_observation must be outside the final JSON object"
        )
    analysis = str(payload.get("_natural_language_analysis", "")).strip()
    subject_items = payload.get("subjects_upsert")
    candidate_items = payload.get("candidates_upsert")
    deletes = payload.get("candidate_deletes")
    if not isinstance(subject_items, list) or not isinstance(candidate_items, list):
        raise ValueError("Repository upserts must be lists")
    if not isinstance(deletes, list) or not all(isinstance(item, str) for item in deletes):
        raise ValueError("candidate_deletes must be a string list")
    raw_delete_set = {item.strip() for item in deletes}

    normalized: dict[str, Any] = {
        "subjects_upsert": [],
        "candidates_upsert": [],
        "candidate_deletes": [],
    }
    seen: set[str] = set()
    detection_index = 1
    for field_name, items in (
        ("subjects_upsert", subject_items),
        ("candidates_upsert", candidate_items),
    ):
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"{field_name}[{item_index}] must be an object")
            subject_id = str(item.get("subject_id", "")).strip()
            if not SUBJECT_ID_RE.fullmatch(subject_id):
                raise ValueError(f"Invalid subject_id: {subject_id!r}")
            if subject_id in seen:
                raise ValueError(f"Duplicate repository upsert: {subject_id}")
            seen.add(subject_id)
            description = str(item.get("subject_description", "")).strip()
            event = str(item.get("event", "")).strip()
            if not description:
                raise ValueError(f"Missing subject_description for {subject_id}")
            if field_name == "candidates_upsert" and not event:
                raise ValueError(f"Candidate event is required for {subject_id}")
            is_promotion = field_name == "subjects_upsert" and subject_id in raw_delete_set
            normalized_item = {
                "subject_id": subject_id,
                "subject_description": description,
                "event": event,
                "promotion_reason": "",
                "reference_crop_required": not is_promotion,
                "detection_index": detection_index,
            }
            if is_promotion:
                if EXTERNAL_TIMESTAMP_FIELD in item:
                    raise ValueError(
                        f"Promoted candidate {subject_id} must omit {EXTERNAL_TIMESTAMP_FIELD}"
                    )
            else:
                timestamp = normalize_timestamp_01(
                    external_timestamp(item, subject_id), subject_id, max_timestamp
                )
                normalized_item["reference_frame_timestamp"] = timestamp
                normalized_item["timestamp_sec"] = timestamp
            normalized[field_name].append(normalized_item)
            detection_index += 1

    subject_ids = {item["subject_id"] for item in normalized["subjects_upsert"]}
    delete_set: set[str] = set()
    for raw_subject_id in deletes:
        subject_id = raw_subject_id.strip()
        if not SUBJECT_ID_RE.fullmatch(subject_id):
            raise ValueError(f"Invalid candidate deletion ID: {subject_id!r}")
        if subject_id in delete_set:
            raise ValueError(f"Duplicate candidate deletion: {subject_id}")
        if subject_id not in subject_ids:
            raise ValueError(f"Candidate deletion requires subject promotion: {subject_id}")
        promoted_item = next(
            item for item in normalized["subjects_upsert"] if item["subject_id"] == subject_id
        )
        promoted_item["promotion_reason"] = (
            analysis or "Promoted based on timestamp-stage natural-language analysis."
        )
        delete_set.add(subject_id)
        normalized["candidate_deletes"].append(subject_id)
    validate_subject_id_text_references(normalized, "timestamp proposal response")
    return normalized


def validate_bbox_proposal_response(
    payload: Any,
    expected_indices: set[int],
) -> dict[int, list[int]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("boxes"), list):
        raise ValueError("Bbox proposal response requires a boxes list")
    boxes: dict[int, list[int]] = {}
    for item_index, item in enumerate(payload["boxes"]):
        if not isinstance(item, dict):
            raise ValueError(f"boxes[{item_index}] must be an object")
        detection_index = int(item.get("detection_index", -1))
        if detection_index in boxes:
            raise ValueError(f"Duplicate bbox index: {detection_index}")
        if detection_index not in expected_indices:
            raise ValueError(f"Unexpected bbox index: {detection_index}")
        boxes[detection_index] = media.clamp_bbox_1000(item.get("bounding_box"))
    if set(boxes) != expected_indices:
        raise ValueError(
            "Bbox indices must exactly match requested frames: "
            f"expected={sorted(expected_indices)}, actual={sorted(boxes)}"
        )
    return boxes


def validate_mixed_bbox_timestamp_response(
    payload: Any,
    bbox_indices: set[int],
    timestamp_retry_indices: set[int],
    max_timestamp: float,
    detection_by_index: dict[int, dict[str, Any]],
) -> tuple[dict[int, list[int]], dict[int, float]]:
    if not isinstance(payload, dict):
        raise ValueError("Mixed bbox/timestamp response must be an object")
    boxes = validate_bbox_proposal_response(
        {"boxes": payload.get("boxes")}, bbox_indices
    )
    revisions = payload.get("timestamp_revisions")
    if not isinstance(revisions, list):
        raise ValueError("Mixed bbox/timestamp response requires timestamp_revisions")
    timestamps: dict[int, float] = {}
    for item_index, item in enumerate(revisions):
        if not isinstance(item, dict):
            raise ValueError(f"timestamp_revisions[{item_index}] must be an object")
        detection_index = int(item.get("detection_index", -1))
        if detection_index in timestamps:
            raise ValueError(f"Duplicate timestamp revision index: {detection_index}")
        if detection_index not in timestamp_retry_indices:
            raise ValueError(f"Unexpected timestamp revision index: {detection_index}")
        subject_id = detection_by_index[detection_index]["subject_id"]
        timestamps[detection_index] = normalize_timestamp_01(
            external_timestamp(item, subject_id), subject_id, max_timestamp
        )
    if set(timestamps) != timestamp_retry_indices:
        raise ValueError(
            "Timestamp revision indices must exactly match requested retries: "
            f"expected={sorted(timestamp_retry_indices)}, actual={sorted(timestamps)}"
        )
    return boxes, timestamps


def validate_frame_review_bbox_response(
    payload: Any,
    expected_indices: set[int],
    max_timestamp: float,
    detection_by_index: dict[int, dict[str, Any]],
) -> tuple[dict[int, list[int]], dict[int, float]]:
    if not isinstance(payload, dict):
        raise ValueError("Frame review response must be an object")
    raw_boxes = payload.get("boxes")
    raw_revisions = payload.get("timestamp_revisions")
    if not isinstance(raw_boxes, list):
        raise ValueError("Frame review response requires a boxes list")
    if not isinstance(raw_revisions, list):
        raise ValueError("Frame review response requires timestamp_revisions")
    bbox_indices = {
        int(item.get("detection_index", -1))
        for item in raw_boxes
        if isinstance(item, dict)
    }
    timestamp_retry_indices = {
        int(item.get("detection_index", -1))
        for item in raw_revisions
        if isinstance(item, dict)
    }
    overlap = bbox_indices & timestamp_retry_indices
    if overlap:
        raise ValueError(
            "Frame review indices cannot appear in both boxes and "
            f"timestamp_revisions: {sorted(overlap)}"
        )
    actual_indices = bbox_indices | timestamp_retry_indices
    if actual_indices != expected_indices:
        raise ValueError(
            "Frame review indices must exactly match requested frames: "
            f"expected={sorted(expected_indices)}, actual={sorted(actual_indices)}"
        )
    return validate_mixed_bbox_timestamp_response(
        payload,
        bbox_indices,
        timestamp_retry_indices,
        max_timestamp,
        detection_by_index,
    )


def validate_crop_review_response(
    payload: Any,
    expected_indices: set[int],
    max_timestamp: float,
    detection_by_index: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Crop review response must be an object")
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("Crop review response requires a reviews list")
    review_by_index: dict[int, dict[str, Any]] = {}
    for item_index, item in enumerate(reviews):
        if not isinstance(item, dict):
            raise ValueError(f"reviews[{item_index}] must be an object")
        detection_index = int(item.get("detection_index", -1))
        if detection_index in review_by_index:
            raise ValueError(f"Duplicate crop review index: {detection_index}")
        if detection_index not in expected_indices:
            raise ValueError(f"Unexpected crop review index: {detection_index}")
        accepted = item.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError(
                f"accepted must be boolean for crop review {detection_index}"
            )
        update_reference = item.get("update_reference")
        if not isinstance(update_reference, bool):
            raise ValueError(
                f"update_reference must be boolean for crop review {detection_index}"
            )
        update_reference_reason = str(
            item.get("update_reference_reason", "")
        ).strip()
        if not update_reference_reason:
            raise ValueError(
                "Missing update_reference reason for crop review "
                f"{detection_index}"
            )
        review = {
            "detection_index": detection_index,
            "accepted": accepted,
            "update_reference": update_reference,
            "update_reference_reason": update_reference_reason,
        }
        if accepted:
            if EXTERNAL_TIMESTAMP_FIELD in item or "bounding_box" in item:
                raise ValueError(
                    "Accepted crop reviews must not include corrected crop "
                    f"parameters for index {detection_index}"
                )
        else:
            if update_reference:
                raise ValueError(
                    f"Rejected crop review cannot update reference: {detection_index}"
                )
            subject_id = detection_by_index[detection_index]["subject_id"]
            review["reference_frame_timestamp"] = normalize_timestamp_01(
                external_timestamp(item, subject_id), subject_id, max_timestamp
            )
            review["bounding_box"] = media.clamp_bbox_1000(item.get("bounding_box"))
        review_by_index[detection_index] = review
    if set(review_by_index) != expected_indices:
        raise ValueError(
            "Crop review indices must exactly match requested crops: "
            f"expected={sorted(expected_indices)}, actual={sorted(review_by_index)}"
        )
    return review_by_index


def validate_mixed_crop_frame_review_response(
    payload: Any,
    crop_indices: set[int],
    frame_indices: set[int],
    max_timestamp: float,
    detection_by_index: dict[int, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]], dict[int, float]]:
    if not isinstance(payload, dict):
        raise ValueError("Mixed crop/frame review response must be an object")
    crop_reviews_payload = {"reviews": payload.get("crop_reviews")}
    crop_reviews = validate_crop_review_response(
        crop_reviews_payload,
        crop_indices,
        max_timestamp,
        detection_by_index,
    )
    boxes, timestamps = validate_frame_review_bbox_response(
        {
            "boxes": payload.get("boxes"),
            "timestamp_revisions": payload.get("timestamp_revisions"),
        },
        frame_indices,
        max_timestamp,
        detection_by_index,
    )
    return crop_reviews, boxes, timestamps


def review_patch_payload(review_payload: dict[str, Any]) -> dict[str, Any]:
    converted = deepcopy(review_payload)
    converted["reviews"] = [
        {
            "reason": review.get(
                "update_reference_reason",
                "Crop accepted by gallery review.",
            ),
            **review,
        }
        for review in converted.get("reviews", [])
    ]
    return converted


def validate_review_response(
    review_payload: Any,
    expected_indices: set[int],
) -> dict[int, dict[str, Any]]:
    if not isinstance(review_payload, dict):
        raise ValueError("Review response must be an object")
    reviews = review_payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("Review response requires a reviews list")
    review_by_index: dict[int, dict[str, Any]] = {}
    for item_index, item in enumerate(reviews):
        if not isinstance(item, dict):
            raise ValueError(f"reviews[{item_index}] must be an object")
        detection_index = int(item.get("detection_index", -1))
        if detection_index in review_by_index:
            raise ValueError(f"Duplicate review index: {detection_index}")
        reason = str(item.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"Missing review reason for index {detection_index}")
        accepted = item.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError(
                f"accepted must be boolean for review index {detection_index}"
            )
        update_reference = item.get("update_reference")
        if not isinstance(update_reference, bool):
            raise ValueError(
                f"update_reference must be boolean for review index {detection_index}"
            )
        update_reference_reason = str(
            item.get("update_reference_reason", "")
        ).strip()
        if not update_reference_reason:
            raise ValueError(
                "Missing update_reference reason for index "
                f"{detection_index}"
            )
        review_by_index[detection_index] = {
            "detection_index": detection_index,
            "accepted": accepted,
            "reason": reason,
            "update_reference": update_reference,
            "update_reference_reason": update_reference_reason,
        }
    if set(review_by_index) != expected_indices:
        raise ValueError(
            "Review indices must exactly match requested crops: "
            f"expected={sorted(expected_indices)}, actual={sorted(review_by_index)}"
        )
    return review_by_index


# Gallery updates and candidate promotion


def flatten_detections(repository_patch: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *repository_patch["subjects_upsert"],
        *repository_patch["candidates_upsert"],
    ]


def visual_review_indices(
    repository: dict[str, Any],
    repository_patch: dict[str, Any],
) -> list[int]:
    delete_set = set(repository_patch["candidate_deletes"])
    indices = []
    for item in flatten_detections(repository_patch):
        subject_id = item["subject_id"]
        crop_required = bool(item.get("reference_crop_required", True))
        if not crop_required:
            candidate = repository["candidates"].get(subject_id)
            if subject_id not in delete_set or candidate is None:
                raise ValueError(
                    "reference_crop_required=false is only valid for an "
                    f"existing candidate promotion: {subject_id}"
                )
            if not candidate.get("reference_image_path"):
                raise ValueError(
                    f"Promotion without a crop requires a saved reference: {subject_id}"
                )
            continue
        indices.append(int(item["detection_index"]))
    return indices


def _index_upserts(items: list[Any], field_name: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{index}] must be an object")
        subject_id = str(item.get("subject_id", "")).strip()
        if not SUBJECT_ID_RE.fullmatch(subject_id):
            raise ValueError(f"Invalid repository ID: {subject_id!r}")
        if subject_id in indexed:
            raise ValueError(f"Duplicate {subject_id} in {field_name}")
        indexed[subject_id] = item
    return indexed


def normalize_entry(
    subject_id: str,
    patch: Any,
    old: dict[str, Any] | None,
    clip_index: int,
    crop: dict[str, Any] | None,
    group_name: str,
) -> dict[str, Any]:
    if not SUBJECT_ID_RE.fullmatch(subject_id):
        raise ValueError(f"Invalid repository ID: {subject_id!r}")
    if not isinstance(patch, dict):
        raise ValueError(f"Patch for {subject_id} must be an object")
    description = str(patch.get("subject_description", "")).strip()
    event = str(patch.get("event", "")).strip()
    if not description:
        raise ValueError(f"Missing subject_description for {subject_id}")
    if group_name == "candidates" and not event:
        raise ValueError(f"Candidate event is required for {subject_id}")
    result = deepcopy(old or {})
    result.update(
        {
            "subject_description": description,
            "first_seen_clip": int(result.get("first_seen_clip", clip_index)),
        }
    )
    update_reference = bool(patch.get("update_reference"))
    if not old and not update_reference:
        raise ValueError(f"New identity must update its reference: {subject_id}")
    if update_reference:
        if crop is None:
            raise ValueError(f"Missing reference crop for {subject_id}")
        result.update(
            {
                "reference_frame_timestamp": crop["timestamp_sec"],
                "bounding_box": crop["bbox_1000"],
                "reference_image_path": crop["crop_path"],
                "reference_frame_path": crop["frame_path"],
            }
        )
    if not result.get("reference_image_path"):
        raise ValueError(f"Missing reference image for {subject_id}")
    if group_name == "candidates":
        event_list = list(result.get("event_list", []))
        event_list.append({"clip_index": clip_index, "event": event})
        result["event_list"] = event_list
    else:
        result.pop("event_list", None)
        promotion_reason = str(patch.get("promotion_reason", "")).strip()
        if promotion_reason:
            result["promotion_reason"] = promotion_reason
    return result


def apply_review_patch(
    repository: dict[str, Any],
    repository_patch: dict[str, Any],
    review_payload: Any,
    crops: list[dict[str, Any]],
    clip_index: int,
) -> dict[str, Any]:
    crop_by_index = {
        int(crop["detection_index"]): crop
        for crop in crops
    }
    expected_indices = set(visual_review_indices(repository, repository_patch))
    if set(crop_by_index) != expected_indices:
        raise ValueError(
            "Crop indices must exactly match visual review indices: "
            f"expected={sorted(expected_indices)}, actual={sorted(crop_by_index)}"
        )
    review_by_index = validate_review_response(review_payload, expected_indices)

    subjects = _index_upserts(
        repository_patch["subjects_upsert"], "subjects_upsert"
    )
    candidates = _index_upserts(
        repository_patch["candidates_upsert"], "candidates_upsert"
    )
    updated = deepcopy(repository)
    patch_groups = (("subjects", subjects), ("candidates", candidates))
    accepted_by_id: dict[str, bool] = {}
    for group_name, patches in patch_groups:
        for subject_id, patch in patches.items():
            detection_index = int(patch["detection_index"])
            crop_required = bool(patch.get("reference_crop_required", True))
            crop = crop_by_index.get(detection_index)
            if crop_required:
                if detection_index not in review_by_index:
                    raise ValueError(f"Missing crop review for {subject_id}")
                review = review_by_index[detection_index]
                accepted = review["accepted"]
                update_reference = review["update_reference"]
                if crop is None:
                    raise ValueError(f"Missing crop for {subject_id}")
            else:
                accepted = True
                update_reference = False
            accepted_by_id[subject_id] = accepted
            if not accepted:
                continue
            if crop is not None and subject_id != crop["subject_id"]:
                raise ValueError(f"Crop identity mismatch for {subject_id}")
            old = updated["subjects"].get(subject_id) or updated["candidates"].get(
                subject_id
            )
            reviewed_patch = {
                **patch,
                "update_reference": update_reference,
            }
            updated[group_name][subject_id] = normalize_entry(
                subject_id, reviewed_patch, old, clip_index, crop, group_name
            )
    delete_set = set(repository_patch["candidate_deletes"])
    for subject_id in delete_set:
        if not accepted_by_id[subject_id]:
            continue
        if subject_id not in repository["candidates"]:
            raise ValueError(f"Cannot promote missing candidate: {subject_id}")
        updated["candidates"].pop(subject_id)
    for subject_id in subjects:
        accepted = accepted_by_id[subject_id]
        if (
            accepted
            and subject_id in repository["candidates"]
            and subject_id not in delete_set
        ):
            raise ValueError(
                f"Promotion requires candidate_deletes entry: {subject_id}"
            )
    overlap = set(updated["subjects"]) & set(updated["candidates"])
    if overlap:
        raise ValueError(f"Repository overlap: {sorted(overlap)}")
    validate_subject_repository(updated, "subject repository")
    return updated


def apply_bbox_response(
    repository_patch: dict[str, Any],
    boxes: dict[int, list[int]],
) -> dict[str, Any]:
    updated = deepcopy(repository_patch)
    for field_name in ("subjects_upsert", "candidates_upsert"):
        for item in updated[field_name]:
            bbox = boxes.get(int(item["detection_index"]))
            if bbox is not None:
                item["bounding_box"] = bbox
                item["bbox_1000"] = bbox
    return updated


def apply_timestamp_revisions(
    repository_patch: dict[str, Any],
    timestamps: dict[int, float],
) -> dict[str, Any]:
    updated = deepcopy(repository_patch)
    for field_name in ("subjects_upsert", "candidates_upsert"):
        for item in updated[field_name]:
            detection_index = int(item["detection_index"])
            if detection_index in timestamps:
                timestamp = timestamps[detection_index]
                item["reference_frame_timestamp"] = timestamp
                item["timestamp_sec"] = timestamp
                item.pop("bounding_box", None)
                item.pop("bbox_1000", None)
    return updated


def apply_crop_review_revisions(
    repository_patch: dict[str, Any],
    rejected_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = deepcopy(repository_patch)
    review_by_index = {
        int(review["detection_index"]): review for review in rejected_reviews
    }
    for field_name in ("subjects_upsert", "candidates_upsert"):
        for item in updated[field_name]:
            review = review_by_index.get(int(item["detection_index"]))
            if review is None:
                continue
            timestamp = review["reference_frame_timestamp"]
            bbox = review["bounding_box"]
            item["reference_frame_timestamp"] = timestamp
            item["timestamp_sec"] = timestamp
            item["bounding_box"] = bbox
            item["bbox_1000"] = bbox
    return updated


def count_crop_review_timestamp_revisions(
    repository_patch: dict[str, Any],
    rejected_reviews: list[dict[str, Any]],
) -> int:
    detection_by_index = {
        int(item["detection_index"]): item for item in flatten_detections(repository_patch)
    }
    count = 0
    for review in rejected_reviews:
        detection_index = int(review["detection_index"])
        original = detection_by_index.get(detection_index)
        if original is None:
            continue
        old_timestamp = round(float(original["timestamp_sec"]), 1)
        new_timestamp = round(float(review["reference_frame_timestamp"]), 1)
        if new_timestamp != old_timestamp:
            count += 1
    return count


def filter_repository_patch_by_indices(
    repository_patch: dict[str, Any],
    indices: set[int],
) -> dict[str, Any]:
    filtered = deepcopy(repository_patch)
    for field_name in ("subjects_upsert", "candidates_upsert"):
        filtered[field_name] = [
            item
            for item in filtered[field_name]
            if int(item["detection_index"]) in indices
        ]
    subject_ids = {
        item["subject_id"] for item in filtered.get("subjects_upsert", [])
    }
    filtered["candidate_deletes"] = [
        subject_id
        for subject_id in filtered.get("candidate_deletes", [])
        if subject_id in subject_ids
    ]
    return filtered


def remove_repository_patch_indices(
    repository_patch: dict[str, Any],
    indices: set[int],
) -> dict[str, Any]:
    remove_ids = {
        item["subject_id"]
        for item in flatten_detections(repository_patch)
        if int(item["detection_index"]) in indices
    }
    remaining = deepcopy(repository_patch)
    for field_name in ("subjects_upsert", "candidates_upsert"):
        remaining[field_name] = [
            item
            for item in remaining[field_name]
            if int(item["detection_index"]) not in indices
        ]
    remaining["candidate_deletes"] = [
        subject_id
        for subject_id in remaining.get("candidate_deletes", [])
        if subject_id not in remove_ids
    ]
    return remaining


def immediate_promotion_indices(repository_patch: dict[str, Any]) -> set[int]:
    return {
        int(item["detection_index"])
        for item in repository_patch.get("subjects_upsert", [])
        if not bool(item.get("reference_crop_required", True))
    }


def apply_immediate_promotions(
    repository: dict[str, Any],
    repository_patch: dict[str, Any],
    clip_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    indices = immediate_promotion_indices(repository_patch)
    if not indices:
        return repository, repository_patch
    promotion_patch = filter_repository_patch_by_indices(repository_patch, indices)
    repository_after = apply_review_patch(
        repository,
        promotion_patch,
        {"reviews": []},
        [],
        clip_index,
    )
    remaining_patch = remove_repository_patch_indices(repository_patch, indices)
    return repository_after, remaining_patch
