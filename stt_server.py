# stt_server.py
import os, tempfile, subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# ---- CONFIG ----
ORIGINS = [
    "*"
]

# Pick a model size you can run:
# "tiny" (fastest), "base", "small", "medium", "large-v3"
MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
# int8 is fast on CPU; change to "float16" if you have GPU
model = WhisperModel(MODEL_NAME, compute_type="int8")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Save upload to temp
    src_suffix = os.path.splitext(audio.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=src_suffix) as f:
        f.write(await audio.read())
        src_path = f.name

    # Convert to WAV mono 16k with ffmpeg (handles webm/mp4/aac/etc.)
    wav_path = src_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
    except Exception:
        try:
            os.unlink(src_path)
        except Exception:
            pass
        return {"text": ""}

    # Transcribe
    segments, info = model.transcribe(wav_path, language="en")
    text = " ".join(seg.text.strip() for seg in segments).strip()

    # Cleanup
    try:
        os.unlink(src_path)
        os.unlink(wav_path)
    except Exception:
        pass

    return {"text": text}
