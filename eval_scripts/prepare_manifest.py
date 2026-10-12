#!/usr/bin/env python3
"""Map public benchmark IDs to local video paths before caption generation."""
import argparse
import json
from pathlib import Path
from pigeons_eval.common import read_jsonl, unique_rows, write_jsonl, resolve_annotations, add_benchmark_argument


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_benchmark_argument(p, annotations=True)
    p.add_argument('--video-root',type=Path,required=True,help='CapRiCorn dataset root, or UGC/WorldSense directory of <video_id>.mp4 files.')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    resolve_annotations(args,p)
    root=args.video_root.resolve()
    if args.benchmark=='capricorn':
        annotations=unique_rows(read_jsonl(args.annotations),'video_path')
        rows=[{'video_path':str((root/key).resolve()),'video_id':key} for key in annotations]
    else:
        data=json.loads(args.annotations.read_text())
        annotations=unique_rows(data['samples'],'video_id') if args.benchmark=='ugc' else data
        rows=[{'video_path':str(root/(key+'.mp4')),'video_id':key} for key in annotations]
    if not rows: raise ValueError('Annotations are empty')
    if len({row['video_path'] for row in rows})!=len(rows): raise ValueError('Ambiguous source paths')
    missing=[row['video_path'] for row in rows if not Path(row['video_path']).is_file()]
    if missing: raise FileNotFoundError(f'{len(missing)} videos are missing, e.g. {missing[:3]}. Supply the matching video directory or build an explicit manifest.')
    write_jsonl(args.output,rows)
    print(f'Wrote {len(rows)} video mappings to {args.output}')
    return 0

if __name__=='__main__':raise SystemExit(main())
