from app.api.youTube_chat import transcribe_with_whisper
import sys

# "Me at the zoo" - Shortest video on YouTube (18s)
# video_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
# video_url = "https://www.youtube.com/watch?v=S0lsXLsDpfs"
video_url = "https://www.youtube.com/watch?v=dY6TDNo2chI"

print(f"Testing Whisper transcription on: {video_url}")
try:
    result = transcribe_with_whisper(video_url)
    print("\nSuccess!")
    print(f"Full text: {result['full_text']}")
    print(f"Chunks: {len(result['chunks'])}")
except Exception as e:
    print(f"\nFailed: {e}")
    sys.exit(1)
