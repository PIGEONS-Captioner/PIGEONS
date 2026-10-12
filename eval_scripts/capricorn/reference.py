#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from math import comb
from typing import Any, Iterable

from pigeons_eval.common import extract_json_from_text, read_jsonl
from . import duration_bin
from pigeons_eval.judge import JudgeConfig, call_with_transport_retries
from .prompts import CATEGORIZE_TEMPLATE


def is_valid_response(model_generation: Any, subject_captions: list[str]) -> bool:
    if not isinstance(model_generation, dict):
        return False
    collected: list[str] = []
    for subject_list in model_generation.values():
        if not isinstance(subject_list, list):
            return False
        collected.extend(subject_list)

    if len(collected) != len(subject_captions):
        if len(subject_captions) < 10:
            return False
        if 10 <= len(subject_captions) < 20 and abs(
            len(collected) - len(subject_captions)
        ) > 1:
            return False
        if 20 <= len(subject_captions) < 30 and abs(
            len(collected) - len(subject_captions)
        ) > 2:
            return False
        if 30 <= len(subject_captions) and abs(
            len(collected) - len(subject_captions)
        ) > 3:
            return False

    if not all(isinstance(value, (str, type(None))) for value in collected):
        return False
    return set(collected) == set(subject_captions)


def process_video(
    stage1_row: dict[str, Any],
    caption: str,
    config: JudgeConfig,
    response_retries: int,
    transport_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "video_path": stage1_row["video_path"],
        "categorization_info": {},
    }
    caption_subjects: dict[str, list[str]] = {}
    for subject_id, event_judgements in stage1_row["main_sbj_info"].items():
        caption_subjects[subject_id] = []
        for item in event_judgements:
            if item["judgement"] in {
                "correctly mentioned",
                "mentioned but with errors",
            }:
                caption_subjects[subject_id].append(
                    item["subject_description_in_caption"]
                )

    for subject_id, subject_captions in caption_subjects.items():
        info: dict[str, Any] = {"sbj_captions": subject_captions}
        if len(subject_captions) > 1:
            prompt = CATEGORIZE_TEMPLATE.format(subject_captions, caption)
            messages = [{"role": "user", "content": prompt}]
            judgement = None
            for attempt in range(response_retries):
                if attempt: time.sleep(retry_delay)
                completion = call_with_transport_retries(
                    messages,
                    config,
                    retries=transport_retries,
                    retry_delay=retry_delay,
                )
                if completion is None:
                    continue
                candidate = extract_json_from_text(completion)
                if is_valid_response(candidate, subject_captions):
                    judgement = candidate
                    break
            info["gpt_categorization"] = (
                judgement if judgement is not None else "Judge Failed"
            )
        else:
            info["gpt_categorization"] = None
        output["categorization_info"][subject_id] = info
    return output


def referential_consistency(counts: Iterable[int], total_events: int) -> float:
    if total_events <= 1:
        raise ValueError("total_events must be greater than 1")
    numerator = sum(comb(count, 2) for count in counts if count >= 2)
    return numerator / comb(total_events, 2)


def calculate_metrics(
    categorization_rows: list[dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    stage1_map = {row["video_path"]: row for row in stage1_rows}
    sample_referential_consistency: list[float] = []
    bins = {
        name: []
        for name in ("(0,120]", "(120,300]", "(300,480]", "(480,601]")
    }
    per_video: list[dict[str, Any]] = []

    for data in categorization_rows:
        metadata = stage1_map[data["video_path"]]
        subject_event_counts = {
            subject_id: len(items)
            for subject_id, items in metadata["main_sbj_info"].items()
        }
        weighted_sum = 0.0
        total_weight = 0
        subject_scores: dict[str, float] = {}

        for subject_id, info in data["categorization_info"].items():
            total_events = subject_event_counts[subject_id]
            if total_events <= 1:
                continue
            categorization = info["gpt_categorization"]
            if categorization == "Judge Failed":
                continue
            if categorization is None:
                score = 0.0
            else:
                counts = [len(items) for items in categorization.values()]
                score = referential_consistency(counts, total_events)
            weighted_sum += score
            total_weight += 1
            subject_scores[subject_id] = score

        if total_weight == 0:
            continue
        video_score = weighted_sum / total_weight
        sample_referential_consistency.append(video_score)
        bucket = duration_bin(metadata["duration"])
        if bucket:
            bins[bucket].append(video_score)
        per_video.append(
            {
                "video_path": data["video_path"],
                "duration": metadata["duration"],
                "referential_consistency": video_score,
                "subject_referential_consistency": subject_scores,
            }
        )

    duration_metrics = {
        name: {
            "count": len(values),
            "referential_consistency": sum(values) / len(values),
        }
        for name, values in bins.items()
        if values
    }
    return {
        "referential_consistency": (
            sum(sample_referential_consistency)
            / len(sample_referential_consistency) if sample_referential_consistency else None
        ),
        "count": len(sample_referential_consistency),
        "duration_bins": duration_metrics,
        "per_video": per_video,
    }

