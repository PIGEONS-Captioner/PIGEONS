"""Validation, checkpointing, and shared evaluation I/O (no model dependencies)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BUNDLED_ANNOTATIONS = {
    'capricorn': 'capricorn/annotations.jsonl',
    'ugc': 'ugc/final_caption_qa.json',
    'worldsense': 'worldsense/worldsense_qa.json',
}


def add_benchmark_argument(parser, *, annotations=False):
    parser.add_argument('--benchmark', choices=BUNDLED_ANNOTATIONS, required=True)
    if annotations:
        parser.add_argument('--annotations', type=Path,
                            help='Defaults to the bundled annotations for the selected benchmark.')


def resolve_annotations(args, parser):
    """Use repository-local annotation copies unless explicitly overridden."""
    if args.annotations is not None:
        return
    if args.benchmark not in BUNDLED_ANNOTATIONS:
        parser.error(f'No bundled annotations for benchmark: {args.benchmark}')
    args.annotations = Path(__file__).resolve().parents[1] / BUNDLED_ANNOTATIONS[args.benchmark]


def resolve_source_path(value, root):
    path = Path(nonempty_text(value, 'source video_path'))
    return str((root / path).resolve())


def load_manifest(path, dataset_root=None, expected_count=None, *, require_files=False):
    """Validate source paths without changing benchmark identifiers in the manifest."""
    if expected_count is not None and expected_count < 1:
        raise ValueError('--expected-count must be positive')
    rows = read_jsonl(path)
    if not rows:
        raise ValueError('Manifest is empty')
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError('Manifest count does not match --expected-count')
    root = (dataset_root or Path(path).resolve().parent).resolve()
    sources = [{'video_path': resolve_source_path(row.get('video_path'), root)}
               for row in rows]
    unique_rows(sources, 'video_path')
    if require_files:
        for row in sources:
            if not Path(row['video_path']).is_file():
                raise FileNotFoundError(row['video_path'])
    return rows, root


def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as handle:
        for number,line in enumerate(handle,1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f'Invalid JSON at {path}:{number}') from exc
            if not isinstance(row,dict): raise ValueError(f'Expected an object at {path}:{number}')
            rows.append(row)
    return rows


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def unique_rows(rows, key):
    result={}
    for row in rows:
        value=row.get(key)
        if not isinstance(value,str) or not value.strip(): raise ValueError(f'Missing/non-string {key}')
        if value in result: raise ValueError(f'Duplicate {key}: {value}')
        result[value]=row
    return result


def nonempty_text(value, label):
    if not isinstance(value,str) or not value.strip(): raise ValueError(f'{label} must be non-empty text')
    return value


def seconds(value):
    if isinstance(value,str): value=value.strip().removesuffix('s')
    if isinstance(value,bool): raise ValueError('Duration must be numeric seconds')
    value=float(value)
    if not math.isfinite(value) or value<=0: raise ValueError('Duration must be positive finite seconds')
    return value


def world_bin(duration):
    for low,high in [(0,120),(120,240),(240,360),(360,480),(480,720)]:
        if low<duration<=high: return f'({low},{high}]'
    return 'outside_paper_bins'


def extract_json_from_text(text):
    if not isinstance(text,str): return None
    candidates=[text.strip()]
    candidates += re.findall(r'```(?:json)?\s*(.*?)\s*```',text,flags=re.S|re.I)
    for candidate in candidates:
        try: return json.loads(candidate)
        except json.JSONDecodeError: pass
    for index,char in enumerate(text):
        if char in '{[':
            try: return json.JSONDecoder().raw_decode(text[index:])[0]
            except json.JSONDecodeError: pass
    return None


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def code_digest():
    root = Path(__file__).resolve().parents[1]
    sources = [path for package in ('pigeons_eval', *BUNDLED_ANNOTATIONS)
               for path in (root / package).rglob('*.py')]
    return digest({str(path.relative_to(root)): path.read_text() for path in sorted(sources)})


def verify_coverage(expected, predictions, ignored=()):
    missing=set(expected)-set(predictions)
    extra=set(predictions)-set(expected)-set(ignored)
    if missing or extra:
        raise ValueError(f'Prediction/annotation mismatch: missing={len(missing)} {sorted(missing)[:3]}, extra={len(extra)} {sorted(extra)[:3]}')


def start_run(output, metadata):
    output=Path(output); path=output/'run.json'
    if path.exists():
        if json.loads(path.read_text())!=metadata:
            raise ValueError('Output directory belongs to different inputs, judge settings, or evaluator code; choose a new --output-dir')
    elif output.exists() and any(output.glob('*.json*')):
        raise ValueError('Output directory contains results without matching run metadata; choose a new --output-dir')
    output.mkdir(parents=True,exist_ok=True)
    write_json(path,metadata)


def run_tasks(tasks, worker, path, workers, retry_failed=False):
    """Append durable records; latest record wins when an explicit retry succeeds.

    Each record is tied to both a logical ID and a hash of the exact task. This
    invalidates dependent reference-clustering results after stage-1 retries.
    """
    tasks=unique_rows(tasks,'id'); path=Path(path); existing={}
    if path.exists():
        for row in read_jsonl(path):
            if row.get('id') not in tasks: raise ValueError(f'Unexpected cached task: {row.get("id")}')
            existing[row['id']]=row
    pending=[]; results={}
    for key,task in tasks.items():
        old=existing.get(key)
        if old and old.get('task_hash')==digest(task) and not (retry_failed and old.get('status')!='ok'):
            results[key]=old
        else: pending.append(task)
    print(f'{path.name}: {len(tasks)} tasks, {len(results)} cached, {len(pending)} pending',flush=True)
    with path.open('a',encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,task):task for task in pending}
        for future in as_completed(futures):
            task=futures[future]
            try: result=future.result()
            except Exception as exc:
                # Do not persist provider errors which may contain credentials or response bodies.
                result={'status':'error','error_type':type(exc).__name__}
            row={**result,'id':task['id'],'task_hash':digest(task)}
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); handle.flush()
            results[task['id']]=row
    return [results[key] for key in tasks]
