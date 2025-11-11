# app_modal.py - Backend SD.Next di Modal.com
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import os

# Konfigurasi Modal
app = modal.App("sdnext-backend")
image = modal.Image.debian_slim().pip_install(
    "torch", "diffusers", "transformers", "accelerate",
    "Pillow", "fastapi", "uvicorn", "pydantic"
).run_commands(
    "apt-get update && apt-get install -y git"
)

class Text2ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.5
    seed: int = -1
    model: str = "stable-diffusion-v1-5"

class ImageResponse(BaseModel):
    image_base64: str
    info: dict

# Download model weights saat build image
def download_models():
    from huggingface_hub import snapshot_download
    snapshot_download("runwayml/stable-diffusion-v1-5")
    snapshot_download("stabilityai/stable-diffusion-xl-base-1.0")

image = image.run_function(download_models)

@app.cls(
    gpu="A100",  # atau "T4" untuk lebih murah
    timeout=300,  # 5 menit timeout
    container_idle_timeout=300,  # idle timeout
    image=image
)
class SDNextModel:
    def __enter__(self):
        # Load model saat container start
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load SD 1.5
        self.pipe_v1 = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        
        # Load SDXL
        self.pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        
        print("✅ Model loaded successfully!")
    
    @modal.method()
    def generate(self, request: Text2ImageRequest):
        try:
            # Pilih model
            pipe = self.pipe_sdxl if "xl" in request.model.lower() else self.pipe_v1
            
            # Set seed
            generator = None
            if request.seed != -1:
                generator = torch.Generator(device=self.device).manual_seed(request.seed)
            
            # Generate image
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=request.cfg_scale,
                generator=generator
            ).images[0]
            
            # Convert to base64
            buffered = BytesIO()
            result.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Return info
            info = {
                "model": request.model,
                "seed": request.seed,
                "size": f"{request.width}x{request.height}",
                "steps": request.steps
            }
            
            return ImageResponse(image_base64=img_str, info=info)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint
web_app = FastAPI()

@web_app.post("/generate")
async def generate_image(request: Text2ImageRequest):
    model = SDNextModel()
    response = model.generate.remote(request)
    return response

@web_app.get("/health")
async def health():
    return {"status": "ok", "service": "sdnext-backend"}

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
