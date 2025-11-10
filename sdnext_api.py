import modal
import os
import subprocess
import threading
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -------- CONFIG --------
APP_NAME = "sdnext-api"
ROOT_DIR = "/root/sdnext"
VOL_NAME = "sdnext-data"
SDNEXT_GIT = "https://github.com/vladmandic/sdnext.git"
GPU_TYPE = "L4"

# -------- Build image --------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands([
        "pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        f"git clone {SDNEXT_GIT} {ROOT_DIR}",
        f"cd {ROOT_DIR} && pip install -r requirements.txt --no-cache-dir",
    ])
    .env({"PYTHONUNBUFFERED": "1"})
)

# ✅ Bagian penting ini yang wajib ada!
app = modal.App(name=APP_NAME, image=image)
