import os, subprocess, threading, time, modal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

# --- Config ---
APP_NAME = "sdnext-api"
ROOT_DIR = "/root/sdnext"
VOL_NAME = "sdnext-data"
SDNEXT_GIT = "https://github.com/vladmandic/sdnext.git"
GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "L4")

# --- Build image (clone repo + install deps) ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands([
        "pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        f"git clone {SDNEXT_GIT} {ROOT_DIR}",
        f"cd {ROOT_DIR} && pip install -r requirements.txt --no-cache-dir",
    ])
)

# --- Modal App ---
app = modal.App(name=APP_NAME, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


@app.function(gpu=GPU_TYPE, timeout=3600, scaledown_window=300, volumes={ROOT_DIR: vol})
@modal.web_server(8000, startup_timeout=600)
def run():
    """Start SD.Next backend API"""
    os.chdir(ROOT_DIR)
    os.makedirs("models/Stable-diffusion", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # --- Jalankan SD.Next di background ---
    def start_sdnext():
        cmd = [
            "python", "webui.py",
            "--listen", "0.0.0.0",
            "--port", "8000",
            "--api",
            "--skip-torch-cuda-test",
            "--disable-safe-unpickle",
            "--no-half",
        ]
        subprocess.Popen(cmd, cwd=ROOT_DIR, env=os.environ.copy())

    threading.Thread(target=start_sdnext, daemon=True).start()
    time.sleep(6)

    # --- Wrapper API biar Modal punya endpoint ---
    app_api = FastAPI(title="SD.Next API")
    app_api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app_api.get("/")
    def root():
        return JSONResponse({
            "status": "ok",
            "message": "SD.Next backend running",
            "internal": "http://127.0.0.1:8000"
        })

    @app_api.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    # --- Print URL Modal ke log ---
    modal_url = os.environ.get("MODAL_SERVER_URL", None)
    print("\n" + "═" * 80)
    print("🚀 SD.Next API launched successfully!")
    if modal_url:
        print(f"🌐 Public URL: {modal_url}")
    else:
        print("🌐 Public URL akan muncul di tab 'Web Endpoints' di dashboard Modal.")
    print("═" * 80 + "\n")

    return app_api
