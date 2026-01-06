from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO
from PIL import Image

router = APIRouter()

# Global variable to hold the pipeline
_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        try:
            model_id = "runwayml/stable-diffusion-v1-5"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Using float16 for GPU to save memory, float32 for CPU
            dtype = torch.float16 if device == "cuda" else torch.float32

            _pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True
            )
            _pipe.to(device)

            # Optional: Add hooks for faster generation if needed
            # _pipe.enable_attention_slicing()

        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {str(e)}")
    return _pipe


class ImageGenerationRequest(BaseModel):
    prompt: str


@router.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    """
    Generate an image based on a text prompt using LOCAL Stable Diffusion.
    """
    try:
        pipe = get_pipeline()

        # Run inference
        # We use a small number of steps for faster generation if on CPU
        # num_inference_steps=20-50 is a good range. Default is 50.
        with torch.inference_mode():
            result = pipe(request.prompt, num_inference_steps=30)
            image = result.images[0]

        # Convert PIL image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "success": True,
            "prompt": request.prompt,
            "image_base64": f"data:image/png;base64,{image_base64}",
            "device": str(pipe.device),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
