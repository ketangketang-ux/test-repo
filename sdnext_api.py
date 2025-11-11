# sdnext_api.py - SD.Next FULL dengan Multi-Model, Img2Img, ControlNet Support
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import base64
from io import BytesIO
import torch
import os
import sys
import time
from PIL import Image

GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "L4")

app = modal.App("sdnext-backend-full")

# Image dengan semua dependencies untuk extensions
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "git", "libgl1-mesa-glx", "libglib2.0-0", "pkg-config", "build-essential",
    "ffmpeg", "libsm6", "libxext6", "libxrender-dev"
).pip_install(
    "torch", "torchvision", "torchaudio",
    "diffusers", "transformers", "accelerate", "safetensors", "einops",
    "opencv-python", "Pillow", "fastapi", "uvicorn", "pydantic",
    "gradio", "psutil", "requests", "numpy", "scipy", "huggingface_hub",
    "controlnet-aux", "realesrgan", "gfpgan", "basicsr", "timm"
).run_commands(
    "pip install --no-deps -q omegaconf webuiapi"
)

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

class ImageToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    image_base64: str
    strength: float = 0.75
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "Euler a"
    model: str = "stable-diffusion-v1-5"

class ModelListResponse(BaseModel):
    models: list

class ImageResponse(BaseModel):
    image_base64: str
    info: dict

# Model mapping
MODEL_URLS = {
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "anything-v5": "Linaqruf/anything-v5.0",
    "dreamshaper": "Lykon/DreamShaper",
    "gpt": "dallinmackay/gpt-4o-mini",
}

@app.cls(
    gpu=GPU_TYPE,
    timeout=600,
    scaledown_window=300,
    image=image,
    force_build=True
)
class SDNextModel:
    def __enter__(self):
        print(f"🚀 Initializing SD.Next Full on {GPU_TYPE}...")
        
        if not os.path.exists("/sdnext"):
            os.system("git clone https://github.com/vladmandic/sdnext.git /sdnext")
        
        sys.path.append("/sdnext")
        os.chdir("/sdnext")
        
        os.system("pip install -r requirements.txt --no-deps -q")
        os.system("pip install -r requirements-extra.txt --no-deps -q")
        
        # Install extensions
        os.system("git clone https://github.com/Mikubill/sd-webui-controlnet.git /sdnext/extensions/controlnet")
        os.system("git clone https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git /sdnext/extensions/upscale")
        
        from modules import paths, shared, sd_models
        
        shared.cmd_opts = type('obj', (object,), {
            'ckpt': None, 'data_dir': '/sdnext', 'models_dir': '/models',
            'no_download': True, 'skip_install': True, 'noprogress': True
        })()
        
        paths.script_path = "/sdnext"
        shared.models_path = "/models"
        
        for dir_name in ["Stable-diffusion", "VAE", "Lora", "ControlNet", "ESRGAN"]:
            os.makedirs(f"/models/{dir_name}", exist_ok=True)
        
        shared.opts.data = {}
        shared.sd_model = None
        
        print("✅ SD.Next Full setup complete!")
    
    def _load_model(self, model_name):
        from modules import shared, sd_models
        
        model_path = MODEL_URLS.get(model_name, "runwayml/stable-diffusion-v1-5")
        
        if not os.path.exists(f"/models/Stable-diffusion/{model_name}"):
            print(f"📥 Downloading {model_name}...")
            os.system(f"huggingface-cli download {model_path} --local-dir /models/Stable-diffusion/{model_name}")
        
        shared.opts.sd_model_checkpoint = f"/models/Stable-diffusion/{model_name}"
        sd_models.load_model()
        self.current_model = model_name
        print(f"✅ Model loaded: {model_name}")
    
    @modal.method()
    def generate_txt2img(self, request: Text2ImageRequest):
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
    
    @modal.method()
    def generate_img2img(self, request: ImageToImageRequest):
        try:
            from modules import shared, devices, sd_samplers
            from modules.processing import StableDiffusionProcessingImg2Img, process_images
            
            if not hasattr(self, 'current_model') or self.current_model != request.model:
                self._load_model(request.model)
            
            devices.torch_gc()
            
            img_data = base64.b64decode(request.image_base64)
            input_image = Image.open(BytesIO(img_data))
            
            p = StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                init_images=[input_image],
                width=request.width,
                height=request.height,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                sampler_name=request.sampler,
                seed=request.seed if request.seed != -1 else -1,
                strength=request.strength,
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
                "size": f"{request.width}x{request.height}",
                "strength": request.strength
            }
            
            return ImageResponse(image_base64=img_base64, info=info)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

web_app = FastAPI()

@web_app.get("/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(models=list(MODEL_URLS.keys()))

@web_app.post("/txt2img", response_model=ImageResponse)
async def txt2img(request: Text2ImageRequest):
    model = SDNextModel()
    return model.generate_txt2img.remote(request)

@web_app.post("/img2img", response_model=ImageResponse)
async def img2img(request: ImageToImageRequest):
    model = SDNextModel()
    return model.generate_img2img.remote(request)

@web_app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": GPU_TYPE,
        "repo": "vladmandic/sdnext",
        "features": ["txt2img", "img2img", "multi-model", "extensions"]
    }

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
