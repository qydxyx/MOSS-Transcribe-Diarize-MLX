import os
import time
import json
import asyncio
import tempfile
import subprocess
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_audio.stt import load

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.transcript_parser import TranscriptStreamParser

app = FastAPI(title="MOSS Transcribe & Diarize Service on Apple Silicon MLX")

MODEL_ID = os.environ.get("MOSS_MODEL_ID", "vanch007/mlx-MOSS-Transcribe-Diarize")

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

def _run_inference(audio_path: str, prompt_text: Optional[str], max_tokens: int):
    t0 = time.time()
    kwargs = {"max_tokens": max_tokens}
    if prompt_text and prompt_text.strip():
        kwargs["prompt"] = prompt_text.strip()

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

    return {
        "text": raw_text,
        "segments": segments,
        "cost_seconds": round(cost, 2),
        "generated_tokens": gen_tokens,
        "tokens_per_second": round(tps, 2),
    }

async def _stream_generator(tmp_path: str, prompt_text: Optional[str], max_tokens: Optional[int], auto_save_prefix: Optional[str] = None):
    m_tokens = max_tokens or 16384
    duration = await asyncio.to_thread(get_audio_duration, tmp_path)
    kwargs = {"max_tokens": m_tokens, "stream": True}
    if prompt_text and prompt_text.strip():
        kwargs["prompt"] = prompt_text.strip()

    parser = TranscriptStreamParser()
    t0 = time.time()
    tokens_count = 0
    last_end_time = 0.0
    all_segments = []

    # 1. 握手启动事件
    yield f"data: {json.dumps({'event': 'start', 'duration': round(duration, 2)})}\n\n"

    try:
        gen = await asyncio.to_thread(model.generate, tmp_path, **kwargs)
        
        while True:
            chunk = await asyncio.to_thread(next, gen, None)
            if chunk is None:
                break
            
            tokens_count += 1
            chunk_text = getattr(chunk, "text", "") or ""
            new_segs = parser.feed(chunk_text)
            
            for seg in new_segs:
                if seg.end > last_end_time:
                    last_end_time = seg.end
                pct = round(min(99.0, (last_end_time / duration * 100.0)), 1) if duration > 0 else 0.0
                seg_dict = {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "speaker": seg.speaker,
                    "text": seg.text.strip().replace("\n", " "),
                }
                all_segments.append(seg_dict)
                payload = {
                    "event": "segment",
                    "progress": pct,
                    "currentTime": round(last_end_time, 2),
                    "totalDuration": round(duration, 2),
                    "segment": seg_dict,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        
        # 2. 解析器末尾收敛
        parser.close()
        remain_segs = parser.feed("")
        for seg in remain_segs:
            last_end_time = max(last_end_time, seg.end)
            seg_dict = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "speaker": seg.speaker,
                "text": seg.text.strip().replace("\n", " "),
            }
            all_segments.append(seg_dict)
            payload = {
                "event": "segment",
                "progress": 100.0,
                "currentTime": round(last_end_time, 2),
                "totalDuration": round(duration, 2),
                "segment": seg_dict,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 服务端持久化落盘
        saved_file = None
        if auto_save_prefix:
            try:
                with open(f"{auto_save_prefix}.txt", "w", encoding="utf-8") as f:
                    for s in all_segments:
                        f.write(f"[{s['start']}s - {s['end']}s] {s['speaker']}: {s['text']}\n")
                with open(f"{auto_save_prefix}.json", "w", encoding="utf-8") as f:
                    json.dump({"segments": all_segments}, f, ensure_ascii=False, indent=2)
                saved_file = f"{auto_save_prefix}.txt"
            except Exception as e_save:
                print(f"Stream auto-save error: {e_save}")

        cost = time.time() - t0
        tps = round(tokens_count / cost, 2) if cost > 0 else 0.0
        yield f"data: {json.dumps({'event': 'complete', 'progress': 100.0, 'cost_seconds': round(cost, 2), 'tokens': tokens_count, 'tps': tps, 'saved_to': saved_file})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "mlx",
        "model": MODEL_ID,
    }

@app.post("/v1/audio/transcriptions")
@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    max_new_tokens: Optional[int] = Form(16384),
    stream: Optional[bool] = Query(False),
):
    # 确保输出目录存在
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
            _stream_generator(tmp_path, prompt, max_new_tokens, auto_save_prefix),
            media_type="text/event-stream"
        )

    try:
        data = await asyncio.to_thread(_run_inference, tmp_path, prompt, max_new_tokens)
        # 服务端兜底自动保存一份
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
