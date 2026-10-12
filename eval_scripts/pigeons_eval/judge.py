"""Public judge clients. No private RPC, embedded keys, or proxy changes."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class JudgeConfig:
    backend: str
    model: str
    base_url: str='https://api.openai.com/v1'
    api_key_env: str='OPENAI_API_KEY'
    timeout: float=120.
    max_tokens: int=3072
    temperature: float=0.
    seed: int=42
    thinking_budget: int=512

    def public_dict(self):
        return asdict(self)


def chat_completion(messages: list[dict[str,Any]], config: JudgeConfig):
    if config.backend=='openai-compatible':
        import requests
        url=config.base_url.rstrip('/')
        if not url.endswith('/chat/completions'):
            if not url.endswith('/v1'): url+='/v1'
            url+='/chat/completions'
        headers={'Content-Type':'application/json'}
        key=os.getenv(config.api_key_env,'')
        if key: headers['Authorization']='Bearer '+key
        response=requests.post(url,headers=headers,json={
            'model':config.model,'messages':messages,'temperature':config.temperature,
            'max_tokens':config.max_tokens,'seed':config.seed,
        },timeout=config.timeout)
        response.raise_for_status()
        choice=response.json()['choices'][0]
        if choice.get('finish_reason') not in (None,'stop'): return None
        value=choice['message']['content']
        return value if isinstance(value,str) and value.strip() else None
    if config.backend=='gemini':
        from google import genai
        from google.genai import types
        # Explicit GOOGLE_GENAI_USE_VERTEXAI=true selects ADC; otherwise use the
        # Gemini developer API with GEMINI_API_KEY (or --api-key-env).
        vertex=os.getenv('GOOGLE_GENAI_USE_VERTEXAI','').lower()=='true'
        kwargs={'http_options':types.HttpOptions(timeout=int(config.timeout*1000))}
        if vertex:
            kwargs.update(vertexai=True,project=os.environ['GOOGLE_CLOUD_PROJECT'],
                          location=os.getenv('GOOGLE_CLOUD_LOCATION','global'))
        else: kwargs['api_key']=os.environ[config.api_key_env]
        system='\n'.join(m['content'] for m in messages if m['role']=='system')
        contents=[types.Content(role='model' if m['role']=='assistant' else 'user',
                  parts=[types.Part.from_text(text=m['content'])]) for m in messages if m['role']!='system']
        with genai.Client(**kwargs) as client:
            response=client.models.generate_content(model=config.model,contents=contents,config=types.GenerateContentConfig(
                system_instruction=system or None,temperature=config.temperature,top_p=.001,
                seed=config.seed,max_output_tokens=config.max_tokens,
                thinking_config=types.ThinkingConfig(include_thoughts=True,thinking_budget=config.thinking_budget),
            ))
        if not response.candidates: return None
        candidate=response.candidates[0]
        if candidate.finish_reason and str(candidate.finish_reason).split('.')[-1]!='STOP': return None
        parts=candidate.content.parts if candidate.content else []
        return '\n'.join(p.text for p in parts if getattr(p,'text',None) and not getattr(p,'thought',False)).strip() or None
    raise ValueError(f'Unknown judge backend: {config.backend}')


def call_with_transport_retries(messages, config, *, retries, retry_delay):
    for attempt in range(retries):
        try:
            response=chat_completion(messages,config)
            if response: return response
        except Exception as exc:
            print(f'Judge request failed ({type(exc).__name__}); attempt {attempt+1}/{retries}',flush=True)
        if attempt+1<retries: time.sleep(retry_delay)
    return None
