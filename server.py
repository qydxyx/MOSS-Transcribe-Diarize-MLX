import os
import re
import gc
import time
import json
import asyncio
import tempfile
import subprocess
from typing import Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import mlx.core as mx
from mlx_audio.stt import load

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.transcript_parser import TranscriptStreamParser

app = FastAPI(title="MOSS Transcribe & Diarize Service on Apple Silicon MLX (VAD Chunked)")

MODEL_ID = os.environ.get("MOSS_MODEL_ID", "vanch007/mlx-MOSS-Transcribe-Diarize")
TARGET_CHUNK_SEC = int(os.environ.get("MOSS_CHUNK_SEC", "600"))  # 默认 10 分钟切片

print(f"[MOSS Server] Initializing MLX backend with model: {MODEL_ID}...")
t_load = time.time()
model = load(MODEL_ID)
print(f"[MOSS Server] MLX Model ready in {time.time() - t_load:.2f}s!")

def get_audio_duration(file_path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        return float(out)
    except Exception:
        return 0.0

def find_vad_splits(audio_path: str, total_dur: float, target_chunk_sec: int = 600, window_sec: int = 60) -> List[float]:
    """使用 ffmpeg silencedetect 在目标切割点附近寻找最自然的静音停顿"""
    if total_dur <= target_chunk_sec * 1.2:
        return [0.0, round(total_dur, 2)]

    splits = [0.0]
    cur = 0.0
    while cur + target_chunk_sec < total_dur:
        win_start = max(cur, cur + target_chunk_sec - window_sec)
        win_end = min(total_dur, cur + target_chunk_sec + window_sec)
        
        cmd = [
            "ffmpeg", "-y", "-ss", str(win_start), "-to", str(win_end),
            "-i", audio_path,
            "-af", "silencedetect=noise=-30dB:d=0.3",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        starts = [float(x) + win_start for x in re.findall(r"silence_start: ([\d\.]+)", res.stderr)]
        ends = [float(x) + win_start for x in re.findall(r"silence_end: ([\d\.]+)", res.stderr)]
        
        best_split = cur + target_chunk_sec
        if starts and ends:
            silences = [(s, e, (s + e) / 2.0) for s, e in zip(starts, ends) if e > s]
            if silences:
                silences.sort(key=lambda x: abs(x[2] - (cur + target_chunk_sec)))
                best_split = round(silences[0][2], 2)

        splits.append(best_split)
        cur = best_split

    splits.append(round(total_dur, 2))
    return sorted(list(set(splits)))

def slice_audio(audio_path: str, start_sec: float, dur_sec: float, out_path: str):
    """无损切片音频，保持相同容器封装"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(dur_sec),
        "-i", audio_path,
        "-c:a", "copy",
        out_path
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def clean_mlx_memory():
    """强制清理 Python 与 MLX 统一显存缓存"""
    gc.collect()
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass

def _run_inference_chunk(audio_path: str, prompt_text: Optional[str], max_tokens: int):
    kwargs = {"max_tokens": max_tokens}
    if prompt_text and prompt_text.strip():
        kwargs["prompt"] = prompt_text.strip()

    t0 = time.time()
    res = model.generate(audio_path, **kwargs)
    cost = time.time() - t0

    raw_text = getattr(res, "text", "") or ""
    segments = []
    if hasattr(res, "segments") and res.segments:
        for s in res.segments:
            segments.append({
                "start": round(float(s.get("start", 0.0)), 2),
                "end": round(float(s.get("end", 0.0)), 2),
                "speaker": s.get("speaker_id", ""),
                "text": s.get("text", "").strip().replace("\n", " "),
            })
    else:
        parsed = parse_transcript(raw_text)
        segments = [
            {
                "start": round(float(s.start), 2),
                "end": round(float(s.end), 2),
                "speaker": s.speaker,
                "text": s.text.strip().replace("\n", " "),
            }
            for s in parsed
        ]

    gen_tokens = getattr(res, "generation_tokens", 0) or 0
    tps = getattr(res, "generation_tps", 0.0) or 0.0
    return raw_text, segments, cost, gen_tokens, tps

def _run_pipeline(audio_path: str, prompt_text: Optional[str], max_tokens: Optional[int]):
    m_tokens = max_tokens or 16384
    duration = get_audio_duration(audio_path)
    splits = find_vad_splits(audio_path, duration, target_chunk_sec=TARGET_CHUNK_SEC)
    
    all_segments = []
    full_text_parts = []
    total_tokens = 0
    t0 = time.time()

    prev_context = prompt_text or ""

    for i in range(len(splits) - 1):
        c_start = splits[i]
        c_end = splits[i + 1]
        c_dur = c_end - c_start
        if c_dur <= 0.1:
            continue

        ext = os.path.splitext(audio_path)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as chunk_tmp:
            chunk_path = chunk_tmp.name

        try:
            slice_audio(audio_path, c_start, c_dur, chunk_path)
            raw_text, segs, cost, tokens, tps = _run_inference_chunk(chunk_path, prev_context, m_tokens)
            total_tokens += tokens
            full_text_parts.append(raw_text)

            # 加上时间基准偏移
            for s in segs:
                s["start"] = round(s["start"] + c_start, 2)
                s["end"] = round(s["end"] + c_start, 2)
                all_segments.append(s)

            # 取最后两句话作为下一个 chunk 的上下文提示
            if segs:
                prev_context = " ".join([s["text"] for s in segs[-2:]])
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            clean_mlx_memory()

    total_cost = round(time.time() - t0, 2)
    overall_tps = round(total_tokens / total_cost, 2) if total_cost > 0 else 0.0

    return {
        "text": "".join(full_text_parts),
        "segments": all_segments,
        "cost_seconds": total_cost,
        "generated_tokens": total_tokens,
        "tokens_per_second": overall_tps,
        "chunks_processed": len(splits) - 1,
    }

async def _stream_pipeline_generator(tmp_path: str, prompt_text: Optional[str], max_tokens: Optional[int], auto_save_prefix: Optional[str] = None):
    m_tokens = max_tokens or 16384
    duration = await asyncio.to_thread(get_audio_duration, tmp_path)
    splits = await asyncio.to_thread(find_vad_splits, tmp_path, duration, TARGET_CHUNK_SEC)
    num_chunks = len(splits) - 1

    yield f"data: {json.dumps({'event': 'start', 'duration': round(duration, 2), 'chunks': num_chunks})}\n\n"

    t0 = time.time()
    total_tokens = 0
    all_segments = []
    prev_context = prompt_text or ""

    try:
        for idx in range(num_chunks):
            c_start = splits[idx]
            c_end = splits[idx + 1]
            c_dur = c_end - c_start
            if c_dur <= 0.1:
                continue

            ext = os.path.splitext(tmp_path)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as chunk_tmp:
                chunk_path = chunk_tmp.name

            try:
                await asyncio.to_thread(slice_audio, tmp_path, c_start, c_dur, chunk_path)

                # 使用流式解析器解析当前 chunk
                parser = TranscriptStreamParser()
                chunk_kwargs = {"max_tokens": m_tokens, "stream": True}
                if prev_context and prev_context.strip():
                    chunk_kwargs["prompt"] = prev_context.strip()

                gen = await asyncio.to_thread(model.generate, chunk_path, **chunk_kwargs)
                chunk_segs = []

                while True:
                    chunk_res = await asyncio.to_thread(next, gen, None)
                    if chunk_res is None:
                        break
                    total_tokens += 1
                    token_text = getattr(chunk_res, "text", "") or ""
                    new_segs = parser.feed(token_text)
                    for seg in new_segs:
                        seg_dict = {
                            "start": round(seg.start + c_start, 2),
                            "end": round(seg.end + c_start, 2),
                            "speaker": seg.speaker,
                            "text": seg.text.strip().replace("\n", " "),
                        }
                        all_segments.append(seg_dict)
                        chunk_segs.append(seg_dict)
                        pct = round(min(99.0, (seg_dict["end"] / duration * 100.0)), 1) if duration > 0 else 0.0
                        payload = {
                            "event": "segment",
                            "progress": pct,
                            "currentTime": seg_dict["end"],
                            "totalDuration": round(duration, 2),
                            "chunk": f"{idx + 1}/{num_chunks}",
                            "segment": seg_dict,
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                parser.close()
                for seg in parser.feed(""):
                    seg_dict = {
                        "start": round(seg.start + c_start, 2),
                        "end": round(seg.end + c_start, 2),
                        "speaker": seg.speaker,
                        "text": seg.text.strip().replace("\n", " "),
                    }
                    all_segments.append(seg_dict)
                    chunk_segs.append(seg_dict)
                    pct = round(min(99.0, (seg_dict["end"] / duration * 100.0)), 1) if duration > 0 else 0.0
                    payload = {
                        "event": "segment",
                        "progress": pct,
                        "currentTime": seg_dict["end"],
                        "totalDuration": round(duration, 2),
                        "chunk": f"{idx + 1}/{num_chunks}",
                        "segment": seg_dict,
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                if chunk_segs:
                    prev_context = " ".join([s["text"] for s in chunk_segs[-2:]])
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                # 关键：每跑完一个 chunk，彻底释放一次 MLX/Metal 统一显存
                clean_mlx_memory()

            # 每跑完一个分块，立即追加落盘，保证中途断开数据也不丢
            if auto_save_prefix:
                try:
                    with open(f"{auto_save_prefix}.txt", "w", encoding="utf-8") as f:
                        for s in all_segments:
                            f.write(f"[{s['start']}s - {s['end']}s] {s['speaker']}: {s['text']}\n")
                    with open(f"{auto_save_prefix}.json", "w", encoding="utf-8") as f:
                        json.dump({"segments": all_segments}, f, ensure_ascii=False, indent=2)
                except Exception as e_save:
                    print(f"Incremental auto-save error: {e_save}")

        # 每跑完一个分块，立即追加保存中间结果，防止长任务意外中断
        if auto_save_prefix:
            try:
                with open(f"{auto_save_prefix}.txt", "w", encoding="utf-8") as f:
                    for s in all_segments:
                        f.write(f"[{s['start']}s - {s['end']}s] {s['speaker']}: {s['text']}\n")
                with open(f"{auto_save_prefix}.json", "w", encoding="utf-8") as f:
                    json.dump({"segments": all_segments}, f, ensure_ascii=False, indent=2)
            except Exception as e_save:
                print(f"Incremental auto-save error: {e_save}")

        # 全部分片完成，持久化保存
        saved_file = f"{auto_save_prefix}.txt" if auto_save_prefix else None

        total_cost = round(time.time() - t0, 2)
        overall_tps = round(total_tokens / total_cost, 2) if total_cost > 0 else 0.0
        yield f"data: {json.dumps({'event': 'complete', 'progress': 100.0, 'cost_seconds': total_cost, 'tokens': total_tokens, 'tps': overall_tps, 'saved_to': saved_file})}\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        clean_mlx_memory()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "mlx-vad-chunked",
        "model": MODEL_ID,
        "chunk_sec": TARGET_CHUNK_SEC,
    }

@app.post("/v1/audio/transcriptions")
@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    max_new_tokens: Optional[int] = Form(16384),
    stream: Optional[bool] = Query(False),
):
    save_dir = os.path.expanduser("~/Documents/transcripts")
    os.makedirs(save_dir, exist_ok=True)
    orig_name = os.path.splitext(file.filename or "audio")[0]
    time_tag = time.strftime("%Y%m%d_%H%M%S")
    auto_save_prefix = os.path.join(save_dir, f"{orig_name}_{time_tag}")

    suffix = os.path.splitext(file.filename or "audio.m4a")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    if stream:
        return StreamingResponse(
            _stream_pipeline_generator(tmp_path, prompt, max_new_tokens, auto_save_prefix),
            media_type="text/event-stream"
        )

    try:
        data = await asyncio.to_thread(_run_pipeline, tmp_path, prompt, max_new_tokens)
        try:
            with open(f"{auto_save_prefix}.txt", "w", encoding="utf-8") as f:
                for seg in data.get("segments", []):
                    f.write(f"[{seg.get('start')}s - {seg.get('end')}s] {seg.get('speaker')}: {seg.get('text')}\n")
            with open(f"{auto_save_prefix}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            data["saved_to"] = f"{auto_save_prefix}.txt"
        except Exception as save_err:
            print(f"Auto-save error: {save_err}")
        return JSONResponse({"code": 0, "data": data})
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        clean_mlx_memory()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
