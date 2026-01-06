from faster_whisper import WhisperModel
import yt_dlp
import tempfile
import os
import time

_whisper_model = None  # Global cache (model loads only once)

def transcribe_with_faster_whisper(youtube_url: str) -> dict:
    """
    Download audio from YouTube and transcribe it using Faster-Whisper.
    Whisper model loads once globally to improve speed on further calls.
    """

    global _whisper_model
    print(f"Starting Faster-Whisper transcription: {youtube_url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        download_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        try:
            # -------- DOWNLOAD YOUTUBE AUDIO --------
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                file_path = os.path.join(temp_dir, f"{info['id']}.mp3")

            if not os.path.exists(file_path):
                raise Exception("Failed to download audio")

            # -------- LOAD MODEL ONCE (CACHED) --------
            if _whisper_model is None:
                print("Loading Faster-Whisper model first time…")
                _whisper_model = WhisperModel(
                    "tiny",
                    device="cpu",              # GPU → "cuda"
                    compute_type="int8"         # GPU → "float16"
                )
            else:
                print("Using cached model…")

            # -------- TRANSCRIBE AUDIO --------
            print("Transcribing...")
            segments, info = _whisper_model.transcribe(
                file_path,
                beam_size=5,   # higher → better accuracy, slower
                best_of=5
            )

            text = " ".join([seg.text for seg in segments])

            return { "full_text": text }

        except Exception as e:
            print("Transcription failed:", e)
            raise e

