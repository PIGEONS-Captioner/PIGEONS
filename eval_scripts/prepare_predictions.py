#!/usr/bin/env python3
"""Export inference outputs using an explicit benchmark-ID/source-video manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pigeons_eval.common import (
    read_jsonl, write_json, write_jsonl, nonempty_text, add_benchmark_argument,
    load_manifest, resolve_source_path,
)


def truncate_repetition(caption):
    """Research preprocessing: truncate at earliest second occurrence of a
    100/101-character substring occurring at least four times (overlap allowed).
    This is suffix truncation, not sentence-level deduplication.
    """
    if len(caption)<400: return caption
    cut=len(caption)
    for length in (101,100):
        positions={}
        for index in range(len(caption)-length+1):
            piece=caption[index:index+length]
            positions.setdefault(piece,[]).append(index)
        for occurrences in positions.values():
            if len(occurrences)>=4: cut=min(cut,occurrences[1])
    return caption[:cut]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_benchmark_argument(p)
    p.add_argument('--manifest',type=Path,required=True,help='JSONL with source video_path and benchmark identifiers.')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--inference-dir',type=Path,help='PIGEONS output root containing per-video run.json and full_caption.txt.')
    source.add_argument('--captions',type=Path,help='Existing JSONL keyed by source video_path with caption.')
    p.add_argument('--dataset-root',type=Path,help='Resolve relative source paths here; otherwise relative to the manifest.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-count',type=int)
    p.add_argument('--allow-partial',action='store_true',help='Explicitly export available captions from degraded/partial inference runs.')
    p.add_argument('--repetition-policy',choices=['legacy','none'],default='legacy')
    args=p.parse_args(argv)
    allow_subset = args.benchmark == 'capricorn' and args.allow_partial
    manifest, root = load_manifest(args.manifest, args.dataset_root, args.expected_count)
    def source_path(value):
        return resolve_source_path(value, root)
    available={}
    if args.captions:
        for row in read_jsonl(args.captions):
            key=source_path(row.get('video_path'))
            if key in available: raise ValueError(f'Duplicate caption source: {key}')
            caption=row.get('caption')
            if not (allow_subset and (caption is None or isinstance(caption,str) and not caption.strip())):
                nonempty_text(caption,'caption')
            available[key]={'caption':caption,'source_status':'provided'}
    else:
        run_paths=sorted(args.inference_dir.glob('*/run.json'))
        if (args.inference_dir/'run.json').exists(): run_paths.append(args.inference_dir/'run.json')
        for path in run_paths:
            run=json.loads(path.read_text()); key=source_path(run.get('video'))
            if key in available: raise ValueError(f'Multiple inference runs for the same video: {key}')
            summary_path=path.parent/'summary.json'
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            good=summary.get('status')=='completed' and summary.get('repository_status')=='completed'
            caption_path=path.parent/'full_caption.txt'
            available[key]={'caption':caption_path.read_text().strip() if caption_path.exists() else '',
                            'source_status':'completed' if good else 'partial'}
    rows=[]; ids=set(); sources=set(); changed=0; partial=[]; skipped=[]
    for record in manifest:
        source_key=source_path(record.get('video_path'))
        sources.add(source_key)
        if allow_subset and (source_key not in available or not available[source_key]['caption']):
            skipped.append(source_key)
            continue
        if allow_subset and not available[source_key]['caption'].strip():
            skipped.append(source_key)
            continue
        if source_key not in available: raise ValueError(f'Missing caption for manifest source: {source_key}')
        item=available[source_key]
        if item['source_status']=='partial':
            if not args.allow_partial: raise ValueError(f'Incomplete or degraded inference for {source_key}; inspect results or explicitly use --allow-partial')
            partial.append(source_key)
        caption=nonempty_text(item['caption'],'caption')
        normalized=truncate_repetition(caption) if args.repetition_policy=='legacy' else caption
        nonempty_text(normalized,'preprocessed caption')
        changed+=normalized!=caption
        fallback = (record.get('benchmark_video_path', record['video_path'])
                    if args.benchmark == 'capricorn' else Path(record['video_path']).stem)
        value = nonempty_text(record.get('video_id', fallback), 'video_id')
        if value in ids: raise ValueError(f'Duplicate benchmark identifier: {value}')
        ids.add(value)
        rows.append({'video_id':value,'caption':normalized,'source_status':item['source_status'],'repetition_policy':args.repetition_policy})
    if not rows: raise ValueError('No non-empty captions available for export')
    write_jsonl(args.output,rows)
    write_json(args.output.with_suffix(args.output.suffix+'.meta.json'),{
        'benchmark':args.benchmark,'count':len(rows),'repetition_policy':args.repetition_policy,
        'truncated_captions':changed,'partial_sources':partial,
        'skipped_sources':skipped,
        'unused_input_captions':len(set(available)-sources),
    })
    print(f'Exported {len(rows)} predictions; truncated={changed}, partial={len(partial)}')
    return 0

if __name__=='__main__': raise SystemExit(main())
