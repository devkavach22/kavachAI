from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from services.image_generation_chain import generate_image_logic

router = APIRouter()


class ImageGenerationRequest(BaseModel):
    prompt: str

    @validator("prompt")
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v.strip()


class ImageGenerationResponse(BaseModel):
    success: bool
    prompt: str
    image_base64: str
    device: str
    generation_time: str


@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """
    Generate an image based on a text prompt using LOCAL Stable Diffusion.
    """
    try:
        result = generate_image_logic(request.prompt, is_sdxl=False)
        return JSONResponse(
            {"success": True, "prompt": request.prompt, **result},
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-sdxl", response_model=ImageGenerationResponse)
async def generate_image_sdxl(request: ImageGenerationRequest):
    """
    Generate an image based on a text prompt using LOCAL Stable Diffusion XL.
    """
    try:
        result = generate_image_logic(request.prompt, is_sdxl=True)
        return JSONResponse(
            {"success": True, "prompt": request.prompt, **result},
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
