"""Swift 4.2.2 plugin for PIGEONS audiovisual vLLM serving.

Load with --external_plugins. Carries the actual sampled video rate into vLLM
and decodes audio from the shared local source rather than a serialized video.
Only the Qwen3-Omni vLLM video branch is overridden; other modalities retain
Swift's implementation. This is a standalone replacement for internal patches.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from swift.template.templates.qwen import Qwen3OmniTemplate

_original_replace_tag = Qwen3OmniTemplate.replace_tag


def decode_audio(path: str, sample_rate: int) -> np.ndarray:
    result = subprocess.run(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path,
         '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(sample_rate),
         '-f', 'f32le', 'pipe:1'], capture_output=True,
    )
    if result.returncode:
        raise ValueError('PIGEONS audiovisual inference requires a decodable audio track: '
                         + result.stderr.decode(errors='replace'))
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not audio.size:
        raise ValueError(f'Empty audio track: {path}')
    return audio


def replace_tag(self, media_type, index, inputs):
    if media_type != 'video' or self.mode != 'vllm':
        return _original_replace_tag(self, media_type, index, inputs)
    import torch
    from qwen_omni_utils import fetch_video

    objects = inputs.objects or {}
    paths = objects.get('video_paths', [])
    if objects.get('source') != 'deploy' or index >= len(paths):
        raise ValueError('Use the PIGEONS inference client, which supplies video_paths')
    path = str(Path(paths[index]).resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(f'Video must be visible to the Swift server: {path}')
    settings = objects.get('video_settings', [])
    options = settings[index] if index < len(settings) else {}
    video, sample_fps = fetch_video(
        {'video': path, 'fps': options.get('fps', 2.0),
         'max_frames': options.get('max_frames', 240),
         'max_pixels': options.get('max_pixels', 262144)},
        return_video_sample_fps=True,
        image_patch_size=self.processor.image_processor.patch_size,
    )
    if sample_fps <= 0:
        raise ValueError('Invalid sampled video FPS')
    inputs.videos[index] = video.to(torch.uint8) if isinstance(video, torch.Tensor) else video
    inputs.mm_processor_kwargs['do_resize'] = False
    inputs.mm_processor_kwargs['fps'] = sample_fps
    seconds_per_grid = self.processor.image_processor.temporal_patch_size / sample_fps
    inputs.mm_processor_kwargs.setdefault('second_per_grid_ts', []).append(seconds_per_grid)
    if self.use_audio_in_video:
        audio = decode_audio(path, self.sampling_rate)
        inputs.audios.insert(inputs.audio_idx, audio)
        inputs.audio_idx += 1
        inputs.mm_processor_kwargs['use_audio_in_video'] = True
    return ['<|vision_start|><|video_pad|><|vision_end|>']


Qwen3OmniTemplate.replace_tag = replace_tag
