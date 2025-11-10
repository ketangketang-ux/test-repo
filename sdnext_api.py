import os, subprocess, threading, time, modal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Config ---
APP_NAME = "sdnext-api"
ROOT_DIR = "/root/sdnext"  # lokasi repo
DATA_DIR = "/data/sdnext"  # lokasi volume mount
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


@app.function(gpu=GPU_TYPE, timeout=3600, scaledown_window=300, volumes={DATA_DIR: vol})
@modal.web_server(8000, startup_timeout=600)
def run():
    """Start SD.Next backend API"""
    os.makedirs(DATA_DIR, exist_ok=True)

    os.chdir(ROOT_DIR)
    os.makedirs("models/Stable-diffusion", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # --- Jalankan SD.Next ---
    def start_sdnext():
        cmd = [
            "python", "webui.py",
            "--listen", "0.0.0.0",
            "--port", "8000",
            "--api",
            "--skip-torch-cuda-test",
            "--disable-safe-unpickle",
            "--data-dir", DATA_DIR,
        ]
        subprocess.Popen(cmd, cwd=ROOT_DIR, env=os.environ.copy())

    threading.Thread(target=start_sdnext, daemon=True).start()
    time.sleep(6)

    # --- FastAPI wrapper untuk healthcheck ---
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
            "repo": ROOT_DIR,
            "data_volume": DATA_DIR
        })

    @app_api.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    print("🚀 SD.Next API launched successfully!")
    print("🌐 Public URL: check dashboard Web Endpoints on Modal")

    return app_api
