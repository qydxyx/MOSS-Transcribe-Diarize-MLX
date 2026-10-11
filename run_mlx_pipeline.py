#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
from pathlib import Path

os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ.get('PATH', '')

from mlx_audio.stt import load

def get_audio_duration(file_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    return float(subprocess.check_output(cmd, text=True).strip())

def slice_audio(audio_path, start_sec, dur_sec, out_path):
    cmd = ['ffmpeg', '-y', '-ss', str(start_sec), '-t', str(dur_sec), '-i', audio_path, '-c:a', 'copy', out_path]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    audio_path = '/tmp/test_0930_16k.m4a'
    chunk_size = 800
    out_json = '/tmp/mlx_0930_transcript.json'
    out_txt = '/tmp/mlx_0930_transcript.txt'
    
    duration = get_audio_duration(audio_path)
    print(f'[MLX Runner] Target audio: {audio_path}, duration: {duration:.2f}s', flush=True)
    
    t_load = time.time()
    print('[MLX Runner] Loading vanch007/mlx-MOSS-Transcribe-Diarize model...', flush=True)
    model = load('vanch007/mlx-MOSS-Transcribe-Diarize')
    print(f'[MLX Runner] Model loaded in {time.time() - t_load:.2f}s', flush=True)
    
    num_chunks = int(duration // chunk_size) + (1 if duration % chunk_size > 10 else 0)
    all_segments = []
    chunk_stats = []
    
    t_start_total = time.time()
    
    for i in range(num_chunks):
        c_start = i * chunk_size
        c_dur = min(chunk_size, duration - c_start)
        if c_dur <= 0:
            break
            
        c_audio = f'/tmp/mlx_chunk_{i:03d}.m4a'
        slice_audio(audio_path, c_start, c_dur, c_audio)
        
        print(f'[MLX Runner] Chunk {i+1}/{num_chunks} ({c_start:.1f}s - {c_start+c_dur:.1f}s) generating...', flush=True)
        t0 = time.time()
        res = model.generate(c_audio, max_tokens=16384)
        c_cost = time.time() - t0
        
        c_tokens = res.generation_tokens
        c_tps = res.generation_tps
        c_segs = res.segments
        
        print(f'[MLX Runner] Chunk {i+1}/{num_chunks} done in {c_cost:.2f}s | TPS: {c_tps:.2f} | Tokens: {c_tokens} | Segments: {len(c_segs)}', flush=True)
        chunk_stats.append({
            'chunk': i + 1,
            'start': c_start,
            'end': c_start + c_dur,
            'cost_seconds': round(c_cost, 2),
            'tokens': c_tokens,
            'tps': round(c_tps, 2),
            'segments_count': len(c_segs)
        })
        
        for s in c_segs:
            spk = s.get('speaker_id') or s.get('speaker') or 'S01'
            txt = s.get('text', '').strip()
            # Clean text if it still contains [Sxx] prefix
            if txt.startswith(f'[{spk}]'):
                txt = txt[len(f'[{spk}]'):].strip()
            all_segments.append({
                'start': round(s['start'] + c_start, 2),
                'end': round(s['end'] + c_start, 2),
                'speaker': f'C{i+1}_{spk}',
                'text': txt
            })
            
    total_time = time.time() - t_start_total
    total_tokens = sum(c['tokens'] for c in chunk_stats)
    print(f'[MLX Runner] ALL CHUNKS COMPLETE! Total time: {total_time:.2f}s ({total_time/60:.2f}m), Total tokens: {total_tokens}, Total segments: {len(all_segments)}', flush=True)
    
    # Save JSON
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'framework': 'MLX',
            'model': 'vanch007/mlx-MOSS-Transcribe-Diarize',
            'audio_duration': duration,
            'total_time_seconds': round(total_time, 2),
            'total_tokens': total_tokens,
            'segments_count': len(all_segments),
            'chunk_stats': chunk_stats,
            'segments': all_segments
        }, f, ensure_ascii=False, indent=2)
        
    # Save TXT
    lines = []
    for s in all_segments:
        m_s, s_s = divmod(int(s['start']), 60)
        m_e, s_e = divmod(int(s['end']), 60)
        lines.append(f"[{m_s:02d}:{s_s:02d} - {m_e:02d}:{s_e:02d}] {s['speaker']}: {s['text']}")
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    print(f'[MLX Runner] Transcripts saved to {out_json} and {out_txt}', flush=True)

if __name__ == '__main__':
    main()
