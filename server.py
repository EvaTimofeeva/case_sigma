"""
server.py
---------
Простой FastAPI сервис: POST /transcribe { "url": "<audio-url>" } → { "text": "..." }
Запуск:
  uvicorn server:app --host 127.0.0.1 --port 8000
"""
import os
import tempfile
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="T-one ASR Service")

class Req(BaseModel):
    url: HttpUrl
    model: str = "small"
    device: str = "cpu"
    sample_rate: int = 16000

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/transcribe")
def transcribe(body: Req):
    try:
        r = requests.get(str(body.url), stream=True, timeout=40)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix=".audio"); os.close(fd)
        with open(tmp, "wb") as f:
            for ch in r.iter_content(8192):
                if ch: f.write(ch)
        import tone
        text = tone.asr_transcribe(source=tmp, model=body.model, device=body.device, sample_rate=body.sample_rate)
        try: os.remove(tmp)
        except: pass
        return {"text": text or ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
