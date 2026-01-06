from fastapi import APIRouter, File, UploadFile, WebSocket
from typing import Dict
from typing import Optional
import json
import time
import speech_recognition as sr
from vosk import KaldiRecognizer, Model
import json
from llm.faster_whisper import get_fasterwhisper

import os
import tempfile
from faster_whisper import WhisperModel
from services.data_extraction import FileDataExtractor

# Router
router = APIRouter()

file_extractor = FileDataExtractor()

_faster_whisper_model = get_fasterwhisper()


# Testing 1 
def transcribe_audio_bytes_with_faster_whisper(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes data into text using the globally available Faster-Whisper model.
    The audio bytes are written to a temporary file for transcription.
    """
    global _faster_whisper_model

    if _faster_whisper_model is None:
        print("Loading Faster-Whisper model for audio bytes transcription…")
        _faster_whisper_model = get_fasterwhisper()
    else:
        print("Using cached Faster-Whisper model for audio bytes transcription…")

    # Create a temporary file to store the audio bytes
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_bytes)
            temp_file_path = tmp_audio_file.name

        print(f"Transcribing temporary audio file: {temp_file_path}")
        segments, info = _faster_whisper_model.transcribe(
            temp_file_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        transcribed_text = "".join([segment.text for segment in segments])
        print(f"Transcription complete. Language: {info.language}, Text: {transcribed_text[:100]}...")
        return transcribed_text
    except Exception as e:
        print(f"Error during audio bytes transcription: {e}")
        raise
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

# Testing 2
def transcribe_audio_file_with_faster_whisper(audio_file_path: str) -> dict:
    """
    Transcribe an audio file using Faster-Whisper.
    Whisper model loads once globally to improve speed on further calls.
    """

    global _faster_whisper_model
    print(f"Starting Faster-Whisper transcription for: {audio_file_path}")

    try:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        return {"debug": "debug"}
        # -------- LOAD MODEL ONCE (CACHED) --------
        if _faster_whisper_model is None:
            print("Loading Faster-Whisper model first time…")
            _faster_whisper_model = WhisperModel(
                "tiny",
                device="cpu",  # GPU → "cuda"
                compute_type="int8",  # GPU → "float16"
            )
        else:
            print("Using cached model…")

        # -------- TRANSCRIBE AUDIO --------
        print("Transcribing...")
        segments, info = _faster_whisper_model.transcribe(
            audio_file_path,
            beam_size=5,  # higher → better accuracy, slower
            best_of=5,
        )

        text = " ".join([seg.text for seg in segments])

        return {"full_text": text}

    except Exception as e:
        print("Transcription failed:", e)
        raise e



# # Testing 3

# model = Model("vosk-model-small-en-in-0.4")
# recognizer = KaldiRecognizer(model, 16000)

# def speech_recognition_vosk_stream(audio_bytes):
#     if recognizer.AcceptWaveform(audio_bytes):
#         return json.loads(recognizer.Result()).get("text", "")
#     else:
#         return json.loads(recognizer.PartialResult()).get("partial", "")

@router.post("/voice_conversation")
@router.websocket("/voice_conversation")
async def voice_conversation(audio_file: UploadFile = File(...)):
    """
    Endpoint to convert an uploaded audio file (speech) into text.

    Args:
        audio_file (UploadFile): The audio file to be transcribed.

    Returns:
        Dict[str, str]: A dictionary containing the transcribed text.
    """


    audio_bytes = await audio_file.read()
    
    
    # Testing 1  by data extraction
    start_time = time.time()
    audio_result = await file_extractor._extract_audio(audio_bytes,"mp3")
    end_time = time.time()
    print(f"Data extraction took {(end_time - start_time) / 60:.2f} minutes")
    
    # Testing 2 by faster whisper
    start_time = time.time()
    _faster_whisper_result = transcribe_audio_bytes_with_faster_whisper(audio_bytes)
    end_time = time.time()
    print(f"Transcription took {(end_time - start_time) / 60:.2f} minutes")
    
    # Testing 3 by speech recognition
    # audio_result = speech_recognition_google_buffered(audio_bytes)
    return {"debug": "debug","audio_result": audio_result, "faster_whisper_result": _faster_whisper_result}



