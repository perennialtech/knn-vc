import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from knn_vc import load_knn_vc

DATA_DIR = Path(os.getenv("KNN_VC_DATA_DIR", "./data"))
VOICES_DIR = DATA_DIR / "voices"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    model = load_knn_vc(pretrained=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/voices")
def list_voices():
    if not VOICES_DIR.exists():
        return {}
    voices = {}
    for voice_dir in VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            voices[voice_dir.name] = [f.name for f in voice_dir.glob("*.wav")]
    return voices


@app.post("/voices/{voice_id}")
def create_voice_wav(voice_id: str, file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".wav"):
        raise HTTPException(400, "Only .wav files are supported")

    voice_id = Path(voice_id).name
    filename = Path(file.filename).name
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)

    file_path = voice_dir / filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"Added {filename} to voice {voice_id}"}


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str):
    voice_id = Path(voice_id).name
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(404, "Voice not found")
    shutil.rmtree(voice_dir)
    return {"message": f"Deleted voice {voice_id}"}


@app.delete("/voices/{voice_id}/{filename}")
def delete_voice_wav(voice_id: str, filename: str):
    voice_id = Path(voice_id).name
    filename = Path(filename).name
    file_path = VOICES_DIR / voice_id / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    file_path.unlink()

    # Clean up empty voice directories
    if not any((VOICES_DIR / voice_id).iterdir()):
        (VOICES_DIR / voice_id).rmdir()

    return {"message": f"Deleted {filename} from voice {voice_id}"}


def cleanup_files(*paths: str):
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


@app.post("/convert")
def convert(
    voice_id: str = Form(...),
    file: UploadFile = File(...),
):
    voice_id = Path(voice_id).name
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(404, "Voice not found")

    ref_wav_paths = list(voice_dir.glob("*.wav"))
    if not ref_wav_paths:
        raise HTTPException(404, "No reference WAVs found for voice")

    src_fd, src_path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(src_fd, "wb") as f:
        shutil.copyfileobj(file.file, f)

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    try:
        query_seq = model.get_features(src_path)
        matching_set = model.get_matching_set(ref_wav_paths)

        out_wav = model.match(query_seq, matching_set, topk=4)

        torchaudio.save(out_path, out_wav.unsqueeze(0), 16000)
    except Exception as e:
        cleanup_files(src_path, out_path)
        raise HTTPException(500, str(e))

    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename="converted.wav",
        background=BackgroundTask(cleanup_files, src_path, out_path),
    )
