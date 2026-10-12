#!/usr/bin/env python3
"""Run the standalone PIGEONS captioner over an explicit benchmark manifest."""
import argparse
import subprocess
import sys
from pathlib import Path
from pigeons_eval.common import load_manifest, resolve_source_path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,epilog='Additional arguments are forwarded to quick_start/inference.py (e.g. --max-tokens 8192 --overwrite).')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=12346)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--expected-count',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args,extra=parser.parse_known_args(argv)
    if args.workers<1: raise ValueError('--workers must be positive')
    records, root = load_manifest(args.manifest, args.dataset_root, args.expected_count,
                               require_files=True)
    videos = [resolve_source_path(row['video_path'], root) for row in records]
    if any(option=='--video' or option.startswith('--video=') for option in extra): raise ValueError('Use the manifest to specify videos, not --video')
    cmd=[sys.executable,str(Path(__file__).resolve().parents[1]/'quick_start'/'inference.py'),
         '--output-dir',str(args.output_dir),'--host',args.host,'--port',str(args.port),'--workers',str(args.workers)]
    for video in videos: cmd.extend(['--video',str(video)])
    cmd.extend(extra)
    print(f'Validated {len(videos)} videos; output: {args.output_dir}')
    if args.dry_run: return 0
    return subprocess.run(cmd,check=False).returncode

if __name__=='__main__': raise SystemExit(main())
