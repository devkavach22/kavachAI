from faster_whisper import WhisperModel
import yt_dlp
import tempfile
import os
import time

_faster_whisper_model = None  # Global cache (model loads only once)

def get_fasterwhisper():
    global _faster_whisper_model
    if _faster_whisper_model is None:
        # You might want to adjust the model size, device, and compute_type based on your environment
        # For example: "base", "small", "medium", "large-v2" for model size,
        # "cuda" for device if a GPU is available, "float16" for compute_type on GPU.
        _faster_whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return _faster_whisper_model
