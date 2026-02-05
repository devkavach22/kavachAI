import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import base64
from io import BytesIO
import time
import os

# Global variables to hold the pipelines
_pipe = None
_sdxl_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        try:
            model_id = os.getenv("IMAGE_GENERATION_DREAMLIKE", "dreamlike-art/dreamlike-photoreal-2.0")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            _pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True
            )
            _pipe.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {str(e)}")
    return _pipe


def get_sdxl_pipeline():
    global _sdxl_pipe
    if _sdxl_pipe is None:
        try:
            model_id = os.getenv("IMAGE_GENERATION_SDXL", "stabilityai/stable-diffusion-xl-base-1.0")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            variant = "fp16" if device == "cuda" else None
            _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True, variant=variant
            )
            _sdxl_pipe.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load SDXL model: {str(e)}")
    return _sdxl_pipe


def generate_image_logic(prompt: str, is_sdxl: bool = False):
    """
    Core logic for generating an image and converting to base64.
    """
    pipe = get_sdxl_pipeline() if is_sdxl else get_pipeline()

    start_time = time.time()
    with torch.inference_mode():
        # Using 30 steps as balanced default
        result = pipe(prompt, num_inference_steps=30)
        image = result.images[0]
    end_time = time.time()

    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    generation_time_str = f"{minutes}m {seconds}s"

    # Convert PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "image_base64": f"data:image/png;base64,{image_base64}",
        "device": str(pipe.device),
        "generation_time": generation_time_str,
    }
