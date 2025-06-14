import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf
from fastapi.responses import JSONResponse, HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Audio Augmentation")

UPLOAD_DIR = "uploads"
AUGMENT_DIR = "augmented_voices"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUGMENT_DIR, exist_ok=True)

def load_audio(path):
    y, sr = librosa.load(path, sr=None)
    return y, sr

def save_audio(y, sr, path):
    sf.write(path, y, sr)

def change_pitch(y, sr, n_steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def change_speed(y, rate):
    return librosa.effects.time_stretch(y=y, rate=rate)

def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def change_volume(y, gain_db):
    gain_factor = 10 ** (gain_db / 20.0)
    return y * gain_factor

def add_pauses(y, sr, num_pauses=3, pause_duration_ms=300):
    pause = np.zeros(int(sr * pause_duration_ms / 1000))
    split_len = len(y) // (num_pauses + 1)
    parts = []
    for i in range(num_pauses):
        parts.append(y[i * split_len: (i + 1) * split_len])
        parts.append(pause)
    parts.append(y[(num_pauses) * split_len:])
    return np.concatenate(parts)

def generate_augmented_audios(input_path, output_id):
    output_dir = os.path.join(AUGMENT_DIR, output_id)
    os.makedirs(output_dir, exist_ok=True)

    y, sr = load_audio(input_path)

    save_audio(change_pitch(y, sr, 4), sr, os.path.join(output_dir, "pitch_up.wav"))
    save_audio(change_speed(y, 1.75), sr, os.path.join(output_dir, "speed_up.wav"))
    save_audio(add_noise(y, 0.01), sr, os.path.join(output_dir, "noisy.wav"))

    y_quiet = change_volume(y, -25)
    y_paused = add_pauses(y_quiet, sr, 2)
    save_audio(y_paused, sr, os.path.join(output_dir, "volume_pauses.wav"))

    y_pitch_speed = change_speed(change_pitch(y, sr, -3), 0.8)
    save_audio(y_pitch_speed, sr, os.path.join(output_dir, "pitch_speed_down.wav"))

    return output_dir

class UploadResponse(BaseModel):
    message: str
    output_dir: str
    files: List[str]



@app.post("/augment-audio", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav", ".ogg")):
        return JSONResponse(status_code=400, content={"message": "Invalid file format"})

    unique_id = str(uuid4())
    filename = "C:\\infobell\\aitask3.wav"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = generate_augmented_audios(file_path, unique_id)
    output_files = os.listdir(output_path)

    return UploadResponse(
        message="Audio augmented successfully!",
        output_dir=output_path,
        files=output_files
    )
@app.get("/", response_class=HTMLResponse)
async def serve_upload_form():
    with open("index.html", "r") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


@app.get("/style.css")
async def serve_css():
    with open("style.css", "r") as f:
        css = f.read()
    return HTMLResponse(content=css, media_type="text/css")

Instrumentator().instrument(app).expose(app)
