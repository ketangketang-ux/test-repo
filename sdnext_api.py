# sdnext_api.py - COMPATIBLE with Modal v1.0+
import modal
from fastapi import FastAPI, HTTPException  # ✅ FastAPI dari sini
from pydantic import BaseModel
import base64
from io import BytesIO
import torch
import os
import sys
import time

GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "L4")

app = modal.App("sdnext-backend")
image = modal.Image.debian_slim().apt_install(
    "git", "libgl1-mesa-glx", "libglib2.0-0"
).pip_install(
    "torch", "torchvision", "torchaudio",
    extra_options=["--index-url", "https://download.pytorch.org/whl/cu118"]
).pip_install(
    "diffusers", "transformers", "accelerate", "safetensors", "einops",
    "opencv-python", "Pillow", "fastapi", "uvicorn", "pydantic",
    "k-diffusion", "gradio", "psutil", "requests", "numpy", "scipy"
).pip_install("huggingface_hub")

class Text2ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "Euler a"
    model: str = "stable-diffusion-v1-5"

class ImageResponse(BaseModel):
    image_base64: str
    info: dict

@app.cls(
    gpu=GPU_TYPE,
    timeout=600,
    scaledown_window=300,  # ✅ FIX: container_idle_timeout -> scaledown_window
    image=image
)
class SDNextModel:
    def __enter__(self):
        print(f"🚀 Initializing SD.Next on {GPU_TYPE}...")
        
        if not os.path.exists("/sdnext"):
            os.system("git clone https://github.com/vladmandic/sdnext.git /sdnext")
        
        sys.path.append("/sdnext")
        os.chdir("/sdnext")
        
        os.system("pip install -r requirements.txt --no-deps -q")
        os.system("pip install -r requirements-extra.txt --no-deps -q")
        
        from modules import paths, shared, sd_models
        
        shared.cmd_opts = type('obj', (object,), {
            'ckpt': None, 'data_dir': '/sdnext', 'models_dir': '/models',
            'no_download': True, 'skip_install': True
        })()
        
        paths.script_path = "/sdnext"
        shared.models_path = "/models"
        os.makedirs("/models/Stable-diffusion", exist_ok=True)
        print("✅ Setup complete!")
    
    def _load_model(self, model_name):
        from modules import shared, sd_models
        
        model_path = "stabilityai/stable-diffusion-xl-base-1.0" if "xl" in model_name else "runwayml/stable-diffusion-v1-5"
        
        if not os.path.exists(f"/models/Stable-diffusion/{model_name}"):
            print(f"📥 Downloading {model_name}...")
            os.system(f"huggingface-cli download {model_path} --local-dir /models/Stable-diffusion/{model_name}")
        
        shared.opts.sd_model_checkpoint = f"/models/Stable-diffusion/{model_name}"
        sd_models.load_model()
        self.current_model = model_name
        print(f"✅ Model loaded: {model_name}")
    
    @modal.method()
    def generate(self, request: Text2ImageRequest):
        try:
            from modules import shared, devices, sd_samplers
            from modules.processing import StableDiffusionProcessingTxt2Img, process_images
            
            if not hasattr(self, 'current_model') or self.current_model != request.model:
                self._load_model(request.model)
            
            devices.torch_gc()
            
            p = StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                sampler_name=request.sampler,
                seed=request.seed if request.seed != -1 else -1,
                batch_size=1,
            )
            
            processed = process_images(p)
            img = processed.images[0]
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            info = {
                "model": request.model,
                "seed": processed.seed,
                "steps": request.steps,
                "cfg_scale": request.cfg_scale,
                "sampler": request.sampler,
                "size": f"{request.width}x{request.height}"
            }
            
            return ImageResponse(image_base64=img_base64, info=info)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# ✅ FIX: Ganti modal.FastAPI() -> FastAPI()
web_app = FastAPI()

@web_app.post("/generate")
async def generate_image(request: Text2ImageRequest):
    model = SDNextModel()
    return model.generate.remote(request)

@web_app.get("/health")
async def health():
    return {"status": "ok", "gpu": GPU_TYPE, "repo": "vladmandic/sdnext"}

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
