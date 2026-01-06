from fastapi import APIRouter, HTTPException,WebSocket,WebSocketDisconnect
from pydantic import BaseModel, field_validator
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# from llm import wisper_model
# from Project.llm.wisper_model import transcribe_with_faster_whisper
# from Project.llm import wisper_model
from services.youTube_chat_chain import summarize_transcript, answer_transcript
import re
import time
import os
import tempfile
import whisper
import yt_dlp
from faster_whisper import WhisperModel
import yt_dlp
import tempfile
import os
import assemblyai as aai


router = APIRouter()

# _whisper_model = wisper_model.get_whisper_model()


class YoutubeTranscriptChat(BaseModel):
    youtube_link: str
    query: str

    @field_validator("youtube_link")
    @classmethod
    def validate_youtube_link(cls, v):
        if not v or ("youtube" not in v and "youtu.be" not in v):
            raise ValueError("Invalid youtube link")
        return v


def extract_video_id(youtube_url: str) -> str:
    """
    Extract video ID from various YouTube URL formats.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID (Standard video URL)
    - https://www.youtube.com/watch?v=VIDEO_ID&t=60s (With parameters)
    - https://youtu.be/VIDEO_ID (Shortened video link)
    - https://youtu.be/VIDEO_ID?t=60 (Shortened with timestamp)
    - https://www.youtube.com/embed/VIDEO_ID (Embedded player URL)
    - vnd.youtube://VIDEO_ID (Mobile app URL)
    - https://www.youtube.com/shorts/VIDEO_ID (Youtube Shorts)
    - https://www.youtube.com/live/VIDEO_ID (Youtube Live)
    - youtube.com/watch?v=VIDEO_ID (No protocol / shortened form)
    """
    patterns = [
        r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[?&]|$)",  # Standard v=parameter or shortened path
        r"(?:youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/shorts\/|youtube\.com\/live\/)([^&\n?#]+)",
        r"(?:vnd\.youtube:\/\/)([^&\n?#]+)",
        r"youtube\.com\/watch\?.*v=([^&\n?#]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            # Additional check to ensure we didn't match garbage in a query param for the first pattern
            # The first pattern is very broad (v=...), so we rely on ID length (11 chars) usually found in YouTube IDs
            return match.group(1)

    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")


_whisper_model = None


# wisper model
def transcribe_with_whisper(youtube_url: str) -> dict:
    """
    Download audio from YouTube video and transcribe it using Whisper.
    The Whisper model is loaded once and reused for subsequent calls.
    """
    global _whisper_model
    print(f"Starting fallback transcription for: {youtube_url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                video_id = info["id"]
                audio_path = os.path.join(temp_dir, f"{video_id}.mp3")

            if not os.path.exists(audio_path):
                raise Exception("Audio download failed")

            if _whisper_model is None:
                print("Loading Whisper model (first time)...")
                _whisper_model = whisper.load_model("tiny")
            else:
                print("Using pre-loaded Whisper model...")

            print("Transcribing audio...")
            result = _whisper_model.transcribe(audio_path)

            return {
                "full_text": result["text"],
            }

        except Exception as e:
            print(f"Fallback transcription failed: {str(e)}")
            raise e


# faster whisper model
'''
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
'''


@router.post("/youtube_chat/")
async def youtube_chat(youtube_link: str, query: str):
    """
    Fetch YouTube transcript and prepare it for chat/query processing.

    Args:
        youtube_link: YouTube video URL
        query: User query about the video

    Returns:
        JSON response with transcript summary and query answer
    """
    try:
        # Extract video ID from URL
        video_id = extract_video_id(youtube_link)
        # print("Video ID: ", video_id)

        try:
            # Try fetching standard transcript first
            start_time = time.time()
            yt_api = YouTubeTranscriptApi()
            transcript_list = yt_api.fetch(
                video_id
            )  # Using fetch with video_id as per API 1.2.3
            end_time = time.time()
            duration_minutes = (end_time - start_time) / 60
            print(f"Transcript fetched in {duration_minutes:.2f} minutes.")

            # print("transcript_list",transcript_list)

            full_transcript = ""
            for entry in transcript_list.snippets:
                full_transcript += entry.text

            summary = summarize_transcript(full_transcript)

            queryAnswer = answer_transcript(query,full_transcript)

            return {
                "success": True,
                "message": "YouTube transcript fetched successfully",
                "video_id": video_id,
                "video_url": youtube_link,
                "query": query,
                "query_answer": queryAnswer,
                "summary": summary,
                # "full_transcript": full_transcript,
            }

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"Standard transcript unavailable ({str(e)}). Attempting fallback...")
            # Fallback to Whisper

            start_time = time.time()
            transcript_data = transcribe_with_whisper(youtube_link)
            # transcript_data = transcribe_with_faster_whisper(youtube_link)
            end_time = time.time()
            duration_minutes = (end_time - start_time) / 60
            print(f"Whisper transcription took {duration_minutes:.2f} minutes.")

            summary = summarize_transcript(transcript_data["full_text"])

            queryAnswer = answer_transcript(query,transcript_data["full_text"])

            return {
                "success": True,
                "message": "Transcript generated using Whisper fallback",
                "video_id": video_id,
                "video_url": youtube_link,
                "query": query,
                "query_answer": queryAnswer,
                "summary": summary,
                # "transcript": transcript_data,
            }

    except VideoUnavailable:
        raise HTTPException(
            status_code=404, detail="Video is unavailable or does not exist"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching the transcript: {str(e)}",
        )


# assembly ai code
# generate transcript with assembly ai
# Set AssemblyAI API Key
aai.settings.api_key = "be5b53b08155468f87219aed0e5dc9ff"


def download_youtube_audio_tempfile(youtube_url: str) -> str:
    """
    Downloads YouTube audio into a temporary file and returns the file path.
    """
    temp_dir = tempfile.mkdtemp()

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info["id"]

    audio_path = os.path.join(temp_dir, f"{video_id}.mp3")

    if not os.path.exists(audio_path):
        raise Exception("Failed to download YouTube audio")

    return audio_path



def transcribe_audio_with_assemblyai(audio_path: str) -> str:
    """
    Transcribes a local audio file using AssemblyAI.
    """
    config = aai.TranscriptionConfig(speech_models=["universal"])
    transcriber = aai.Transcriber(config=config)

    transcript = transcriber.transcribe(audio_path)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript.text



def transcribe_youtube_with_assemblyai(youtube_url: str) -> dict:
    """
    Full pipeline:
    1. Download YouTube audio as tempfile
    2. Send it to AssemblyAI
    3. Return transcript
    """
    print("Downloading YouTube audio...")
    audio_path = download_youtube_audio_tempfile(youtube_url)

    print("Generating transcript using AssemblyAI...")
    transcript_text = transcribe_audio_with_assemblyai(audio_path)
    print("transcript generated")

    return {
        "transcript": transcript_text
    }



@router.websocket("/youtube_chat")
async def youtube_chat_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "WebSocket ready"})

    # ---------------------- 🔥 FETCH TRANSCRIPT ON CONNECT ---------------------- #
    try:
        data = await websocket.receive_json()

        youtube_link = data.get("youtube_link")
        if not youtube_link:
            await websocket.send_json({"error": "youtube_link is required"})
            return

        await websocket.send_json({"status": "processing", "message": "Extracting video ID"})
        video_id = extract_video_id(youtube_link)

        # Try YouTube transcript first
        try:
            await websocket.send_json({"status": "transcript", "message": "Fetching YouTube transcript..."})

            # comment for assembly ai
            '''
            yt_api = YouTubeTranscriptApi()
            transcript_list = yt_api.fetch(video_id)
            full_transcript = " ".join(entry.text for entry in transcript_list.snippets)
            '''

            # uncomment for assembly ai
            transcript_data = transcribe_youtube_with_assemblyai(youtube_link)
            full_transcript = transcript_data["transcript"]
            summary = summarize_transcript(full_transcript)

            await websocket.send_json({
                "status": "transcript_loaded",
                "via": "youtube_transcript",
                "message": "Transcript loaded successfully",
                "summary": summary
            })

        except (TranscriptsDisabled, NoTranscriptFound):
            await websocket.send_json({"status": "fallback", "message": "Transcript disabled. Using Whisper..."})

            transcript_data = transcribe_with_whisper(youtube_link)
            # transcript_data = transcribe_with_faster_whisper(youtube_link)
            full_transcript = transcript_data["full_text"]
            summary = summarize_transcript(full_transcript)

            await websocket.send_json({
                "status": "transcript_loaded",
                "via": "whisper",
                "message": "Transcript generated with Whisper",
                "summary": summary
            })

    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
        return

    # --------------------------- 🔥 LISTEN FOR LIVE QUERIES --------------------------- #
    try:
        while True:
            request = await websocket.receive_json()
            query = request.get("query")

            if not query:
                await websocket.send_json({"error": "query required"})
                continue

            # Answer user question using stored transcript
            answer = answer_transcript(query, full_transcript)
            

            await websocket.send_json({
                "success": True,
                "answer": answer,
            })

    except WebSocketDisconnect:
        print("WebSocket disconnected by client.")
        # No message can be reliably sent to the client once WebSocketDisconnect occurs,
        # as the connection is already closed from the client's end.
        # Any cleanup or logging can be performed here.
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
        return

