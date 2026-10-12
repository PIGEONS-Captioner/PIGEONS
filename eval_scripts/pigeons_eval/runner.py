"""Shared CLI for the three paper benchmarks."""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

from capricorn import accuracy, reference
from .common import (read_jsonl, write_json, unique_rows, nonempty_text, seconds,
                     world_bin, extract_json_from_text, digest, code_digest,
                     verify_coverage, start_run, run_tasks, resolve_annotations, add_benchmark_argument)
from .judge import JudgeConfig, call_with_transport_retries
from ugc.prompts import SYSTEM_PROMPT as UGC_SYSTEM
from worldsense.prompts import SYSTEM_PROMPT as WORLDSENSE_SYSTEM

UGC_NO_GT='PI-0abb79ae-18ae-40d9-a66f-3ad7744a6095'
EVENT_KEYS=('main_sbj_evts','main_sbj_ints','bg_desc','trans_desc','minor_evts')


def load_inputs(benchmark, annotations_path, predictions_path):
    predictions=unique_rows(read_jsonl(predictions_path),'video_id')
    if not predictions: raise ValueError('Predictions are empty')
    for item in predictions.values(): nonempty_text(item.get('caption'),'caption')
    ignored=[]
    if benchmark=='capricorn':
        rows=read_jsonl(annotations_path)
        annotations=unique_rows(rows,'video_path')
        for row in rows:
            row['duration']=seconds(row['duration'])
            subjects=unique_rows(row['subjects'],'sbj_id')
            for subject in subjects.values(): nonempty_text(subject.get('sbj_desc'),'sbj_desc')
            all_events=[]
            for group in EVENT_KEYS:
                events=row['events'][group]
                if not isinstance(events,list): raise ValueError(f'{group} must be a list')
                for event in events:
                    nonempty_text(event,'event')
                    if group in EVENT_KEYS[:2]:
                        ids=accuracy.extract_subject_ids(event)
                        if not ids or not set(ids)<=set(subjects):
                            raise ValueError('Subject events must reference declared GT subject IDs; use the released annotations')
                    elif event in all_events: raise ValueError('Duplicate event in a non-subject category')
                    all_events.append(event)
            if not all_events: raise ValueError('Annotations must have at least one event per video')
    elif benchmark=='ugc':
        value=json.loads(Path(annotations_path).read_text())
        annotations=unique_rows(value['samples'],'video_id')
        for row in annotations.values(): nonempty_text(row.get('answer'),'GT answer')
        if UGC_NO_GT in predictions and UGC_NO_GT not in annotations: ignored=[UGC_NO_GT]
    else:
        annotations=json.loads(Path(annotations_path).read_text())
        if not isinstance(annotations,dict): raise ValueError('WorldSense annotations must be keyed by video_id')
        for vid,row in annotations.items():
            if not isinstance(row,dict) or row.get('video_id')!=vid: raise ValueError('WorldSense key/video_id mismatch')
            row['duration_seconds']=seconds(row['video_duration'])
            keys=[k for k in row if k.startswith('task')]
            if not keys: raise ValueError(f'No QA tasks for {vid}')
            for task_key in keys:
                task=row[task_key]; nonempty_text(task.get('question'),'question')
                candidates=task.get('candidates')
                if not isinstance(candidates,list) or not 2<=len(candidates)<=26: raise ValueError('Invalid QA candidates')
                for index,choice in enumerate(candidates):
                    nonempty_text(choice,'candidate')
                    if not re.match(rf'^{chr(65+index)}[.)\s]',choice): raise ValueError('Candidates must include ordered A/B/C/... labels')
                if task.get('answer') not in [chr(65+i) for i in range(len(candidates))]: raise ValueError('Invalid GT answer letter')
    if not annotations: raise ValueError('Annotations are empty')
    if benchmark=='capricorn':
        extra=set(predictions)-set(annotations)
        if extra: raise ValueError(f'Predictions contain unknown annotation IDs: {sorted(extra)[:3]}')
        return annotations,predictions,ignored
    verify_coverage(annotations,predictions,ignored)
    return annotations,{k:predictions[k] for k in annotations},ignored


def parse_ugc(text):
    value=extract_json_from_text(text)
    if value is None and isinstance(text,str):
        match=re.search(r'\{.*\}',text,re.S)
        if match:
            try: value=ast.literal_eval(match.group())
            except (ValueError,SyntaxError): pass
    if not isinstance(value,dict) or set(value)!={'visual','audio','details'}: raise ValueError('Invalid UGC score dictionary')
    if not all(type(score) is int and 1<=score<=5 for score in value.values()): raise ValueError('UGC scores must be integers in [1,5]')
    return value


def ugc_task(task, config, retries, delay):
    messages=[{'role':'system','content':UGC_SYSTEM},{'role':'user','content':
              f"Sample ID: {task['id']}\nPredicted Description: {task['caption']}\nGround Truth Description: {task['answer']}\n"}]
    responses=[]
    for attempt in range(retries):
        if attempt: time.sleep(delay)
        raw=call_with_transport_retries(messages,config,retries=1,retry_delay=delay)
        if raw: responses.append(raw)
        try:
            scores=parse_ugc(raw)
            return {'status':'ok','video_id':task['id'],'scores':scores,'responses':responses}
        except ValueError: pass
    return {'status':'failed','video_id':task['id'],'scores':dict.fromkeys(('visual','audio','details'),0),'responses':responses}


def world_task(task, config, retries, delay):
    choices='\n'.join(task['candidates'])
    prompt=f'''\nHere is the video caption:
"{task['caption']}"

Question: {task['question']}
Choices:
    {choices}'''
    raw=call_with_transport_retries([{'role':'system','content':WORLDSENSE_SYSTEM},{'role':'user','content':prompt}],config,retries=retries,retry_delay=delay)
    predicted=raw.strip().upper() if raw else None
    allowed=[chr(65+i) for i in range(len(task['candidates']))]+['N/A']
    status='failed' if raw is None else 'ok' if predicted in allowed else 'invalid'
    return {**task,'status':status,'response':raw,'prediction':predicted,'correct':predicted==task['answer']}


def ugc_metrics(rows):
    count=len(rows)
    # Transport/format failures count as zero, matching the research runner.
    metrics={name:sum(row.get('scores',{}).get(name,0) for row in rows)/count/5*100 for name in ('audio','visual','details')}
    return {'count':count,**metrics,'average':sum(metrics.values())/3,'scale':'0-100',
            'failed_samples':sum(r['status']!='ok' for r in rows)}


def world_metrics(rows):
    bins={}
    for row in rows:
        bucket=world_bin(row['duration_seconds']); bins.setdefault(bucket,[]).append(row)
    return {'count':len(rows),'accuracy':100*sum(r.get('correct',False) for r in rows)/len(rows),'scale':'0-100',
            'unanswered':sum(r.get('prediction')=='N/A' for r in rows),
            'failed_or_invalid':sum(r['status']!='ok' for r in rows),
            'duration_bins':{k:{'count':len(v),'accuracy':100*sum(r.get('correct',False) for r in v)/len(v)} for k,v in bins.items()}}


def parse_args(benchmark=None, argv=None):
    p=argparse.ArgumentParser(description='Evaluate PIGEONS captions against benchmark annotations.')
    if benchmark: p.set_defaults(benchmark=benchmark)
    else: add_benchmark_argument(p)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--annotations',type=Path,help='Defaults to the bundled annotations for the selected benchmark.')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--backend',choices=['openai-compatible','gemini'])
    p.add_argument('--judge-model',help='Defaults: gpt-4.1 (CapRiCorn/UGC), gemini-2.5-pro (WorldSense).')
    p.add_argument('--base-url',default='https://api.openai.com/v1')
    p.add_argument('--api-key-env',help='Environment variable NAME, not the secret; default OPENAI_API_KEY or GEMINI_API_KEY.')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--response-retries',type=int,help='Default: 5/20/10 for CapRiCorn/UGC/WorldSense.')
    p.add_argument('--retry-delay',type=float,default=2.)
    p.add_argument('--timeout',type=float,default=120.)
    p.add_argument('--max-tokens',type=int,default=3072)
    p.add_argument('--temperature',type=float,default=0.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--thinking-budget',type=int,default=512,help='Gemini thinking budget (WorldSense paper setting: 512).')
    p.add_argument('--expected-count',type=int,help='Expected annotation video count; enforces full-release size when specified.')
    p.add_argument('--retry-failed',action='store_true',help='Re-evaluate cached failures; successful cached tasks are retained.')
    p.add_argument('--dry-run',action='store_true',help='Validate inputs and print task counts without API calls or output writes.')
    args=p.parse_args(argv)
    resolve_annotations(args,p)
    return args


def main(benchmark=None, argv=None):
    args=parse_args(benchmark,argv)
    if args.response_retries is None: args.response_retries={'capricorn':5,'ugc':20,'worldsense':10}[args.benchmark]
    for name in ['workers','response_retries','timeout','max_tokens']:
        if getattr(args,name)<=0: raise ValueError(f'--{name.replace("_","-")} must be positive')
    if args.retry_delay<0: raise ValueError('--retry-delay cannot be negative')
    if args.expected_count is not None and args.expected_count<=0: raise ValueError('--expected-count must be positive')
    url=urlsplit(args.base_url)
    if url.scheme not in ('http','https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('--base-url must be an HTTP(S) endpoint without credentials, query or fragment')
    backend=args.backend or ('gemini' if args.benchmark=='worldsense' else 'openai-compatible')
    config=JudgeConfig(backend=backend,model=args.judge_model or ('gemini-2.5-pro' if args.benchmark=='worldsense' else 'gpt-4.1'),
        base_url=args.base_url,api_key_env=args.api_key_env or ('GEMINI_API_KEY' if backend=='gemini' else 'OPENAI_API_KEY'),
        timeout=args.timeout,max_tokens=args.max_tokens,temperature=args.temperature,seed=args.seed,thinking_budget=args.thinking_budget)
    annotations,predictions,ignored=load_inputs(args.benchmark,args.annotations,args.predictions)
    if args.expected_count is not None and len(annotations)!=args.expected_count:
        raise ValueError(f'Expected {args.expected_count} annotation videos; found {len(annotations)}')
    annotation_count=len(annotations)
    if args.benchmark=='capricorn':
        annotations={key:annotations[key] for key in predictions}
    tasks=[]
    for key,row in annotations.items():
        if args.benchmark=='worldsense':
            for name,qa in row.items():
                if name.startswith('task'):
                    tasks.append({'id':f'{key}/{name}','video_id':key,'task_id':name,'duration_seconds':row['duration_seconds'],
                                  'caption':predictions[key]['caption'],**qa})
        elif args.benchmark=='ugc': tasks.append({'id':key,'caption':predictions[key]['caption'],'answer':row['answer']})
        else: tasks.append({'id':key,'prediction':predictions[key],'annotation':row})
    info={'benchmark':args.benchmark,'videos':len(annotations),'tasks':len(tasks),'ignored_without_gt':ignored,'judge':config.public_dict()}
    if args.benchmark=='capricorn':
        info['annotation_videos']=annotation_count
        info['stage1_prompts']=sum(len(r['events'][k]) for r in annotations.values() for k in EVENT_KEYS)
        info['stage2_prompts_depend_on_stage1']=True
    print(json.dumps(info,indent=2),flush=True)
    if args.dry_run: return 0
    if backend=='gemini':
        if os.getenv('GOOGLE_GENAI_USE_VERTEXAI','').lower()=='true':
            if not os.getenv('GOOGLE_CLOUD_PROJECT'): raise ValueError('GOOGLE_CLOUD_PROJECT is required for Vertex AI')
        elif not os.getenv(config.api_key_env): raise ValueError(f'Set {config.api_key_env} before evaluation')
    elif url.hostname=='api.openai.com' and not os.getenv(config.api_key_env): raise ValueError(f'Set {config.api_key_env} before evaluation')
    metadata={**info,'inputs_hash':digest([annotations,predictions]),'evaluator_hash':code_digest(),
              'response_retries':args.response_retries,'retry_delay':args.retry_delay}
    if backend=='gemini':
        from google import genai  # Validate optional dependency before scheduling API work.
    else:
        import requests
    start_run(args.output_dir,metadata)
    run=lambda ts,worker,name:run_tasks(ts,worker,args.output_dir/name,args.workers,args.retry_failed)
    if args.benchmark=='ugc':
        rows=run(tasks,lambda t:ugc_task(t,config,args.response_retries,args.retry_delay),'judgements.jsonl')
        metrics=ugc_metrics(rows)
    elif args.benchmark=='worldsense':
        rows=run(tasks,lambda t:world_task(t,config,args.response_retries,args.retry_delay),'judgements.jsonl')
        # Even worker exceptions remain in QA accuracy's denominator.
        rows=[{**task,**row} for task,row in zip(tasks,rows)]
        metrics=world_metrics(rows)
    else:
        def stage1(task):
            result=accuracy.process_video(task['prediction'],task['annotation'],config,args.response_retries,1,args.retry_delay)
            failed=sum(x['judgement']=='failed' for entries in result['main_sbj_info'].values() for x in entries)+sum(x['judgement']=='failed' for x in result['other_event_judge'])
            return {**result,'status':'failed' if failed else 'ok'}
        rows1=run(tasks,stage1,'1_judge_accuracy_coverage.jsonl')
        valid1=[r for r in rows1 if r['status']!='error']
        tasks2=[{'id':r['id'],'stage1':r,'caption':predictions[r['id']]['caption']} for r in rows1]
        def stage2(task):
            if task['stage1']['status']=='error':
                return {'status':'error','error_type':'UpstreamStageError'}
            result=reference.process_video(task['stage1'],task['caption'],config,args.response_retries,1,args.retry_delay)
            failed=any(i['gpt_categorization']=='Judge Failed' for i in result['categorization_info'].values())
            return {**result,'status':'failed' if failed else 'ok'}
        rows2=run(tasks2,stage2,'2_judge_referential_consistency.jsonl')
        m1=accuracy.calculate_metrics(valid1)
        m2=reference.calculate_metrics([r for r in rows2 if r['status']!='error'],valid1)
        write_json(args.output_dir/'1_accuracy_coverage_metrics.json',m1)
        write_json(args.output_dir/'2_referential_consistency_metrics.json',m2)
        percentage=lambda value:None if value is None else value*100
        metrics={'count':len(annotations),'accuracy':percentage(m1['accuracy']),'coverage':percentage(m1['coverage']),
                 'referential_consistency':percentage(m2['referential_consistency']),'reference_scored_videos':m2['count'],
                 'scale':'0-100','duration_bins':{k:{**{n:percentage(v) if n!='count' else v for n,v in b.items()},
                   'referential_consistency':percentage(m2['duration_bins'].get(k,{}).get('referential_consistency'))} for k,b in m1['duration_bins'].items()}}
        rows=rows1+rows2
    metrics['benchmark']=args.benchmark
    metrics['failed_tasks']=sum(r['status']!='ok' for r in rows)
    metrics['complete']=metrics['failed_tasks']==0
    write_json(args.output_dir/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)
    return 0 if metrics['complete'] else 1
