import os
import io
import json
import zipfile
import rarfile
import tempfile
from typing import Dict, Any, List
import re
import base64
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import docx
import docx2txt
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import HTTPException, UploadFile
import whisper  # Import OpenAI Whisper
import cv2  # OpenCV for video frame extraction


class FileDataExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(["en"])

        # Load Whisper model once (base model is good; you can use "small", "medium", etc.)
        self.whisper_model = whisper.load_model("base",device="cpu")

    async def extract_data(self, file: UploadFile) -> Dict[str, Any]:
        filename = file.filename
        file_bytes = await file.read()
        ext = os.path.splitext(filename)[1].lower()

        extraction_methods = {
            ".pdf": self._extract_pdf,
            ".jpg": self._extract_image,
            ".jpeg": self._extract_image,
            ".png": self._extract_image,
            ".bmp": self._extract_image,
            ".tiff": self._extract_image,
            ".docx": self._extract_docx,
            ".xls": self._extract_excel,
            ".xlsx": self._extract_excel,
            ".xlsm": self._extract_excel,
            ".xltx": self._extract_excel,
            ".xltm": self._extract_excel,
            ".json": self._extract_json,
            ".db": self._extract_sql,
            ".sqlite": self._extract_sql,
            ".sql": self._extract_sql,
            ".zip": self._extract_zip,
            ".rar": self._extract_rar,
            ".html": self._extract_html,
            ".txt": self._extract_txt,
             # 🎧 NEW AUDIO SUPPORT
            ".mp3": self._extract_audio,
            ".wav": self._extract_audio,
            # 🎥 NEW VIDEO SUPPORT
            ".mp4": self._extract_video,
            ".avi": self._extract_video,
            ".mov": self._extract_video,
            ".mkv": self._extract_video,
            ".webm": self._extract_video,
            ".flv": self._extract_video,
            ".wmv": self._extract_video,
        }

        if ext in extraction_methods:
            try:
                return await extraction_methods[ext](file_bytes, ext)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"An error occurred during {ext} extraction: {str(e)}",
                )
        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")

    # =====================================================================================
    # NEW FUNCTION: VIDEO → TEXT (AUDIO + FRAME OCR)
    # =====================================================================================
    async def _extract_video(self, file_bytes: bytes, ext: str):
        """
        Extract text from video using:
        1. Whisper for audio transcription
        2. OCR on key frames for on-screen text
        """
        try:
            # 1. Save video bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file_bytes)
                video_path = tmp.name

            # 2. Audio transcription with Whisper
            audio_text = ""
            try:
                result = self.whisper_model.transcribe(video_path, fp16=False)
                audio_text = result.get("text", "").strip()
            except Exception as audio_err:
                print(f"Audio transcription failed: {audio_err}")

            # 3. Extract text from video frames using OCR
            frame_text = await self._extract_text_from_video_frames(video_path)

            # 4. Cleanup
            os.remove(video_path)

            return {
                "type": "video",
                "format": ext,
                "audio_text": audio_text,
                "frame_text": frame_text,
                "combined_text": f"{audio_text} {frame_text}".strip()
            }

        except Exception as e:
            raise Exception(f"Video extraction failed: {e}")

    async def _extract_text_from_video_frames(self, video_path: str) -> str:
        """
        Extract OCR text from key frames of the video.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frame_texts = []
            frame_count = 0
            keyframe_interval = 30  # Process every 30th frame (adjustable)

            while cap.isOpened() and frame_count < 150:  # Limit to ~5 seconds at 30fps
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % keyframe_interval == 0:
                    # Convert BGR to RGB for OCR
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image for easyocr
                    pil_image = Image.fromarray(frame_rgb)
                    with io.BytesIO() as buf:
                        pil_image.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    
                    # OCR
                    result = self.reader.readtext(img_bytes, detail=0, paragraph=True)
                    frame_texts.append(" ".join(result))

                frame_count += 1

            cap.release()
            return " ".join(frame_texts)

        except Exception as e:
            return f"Frame extraction failed: {e}"

    # =====================================================================================
    # NEW FUNCTION: MP3 / WAV → TEXT USING WHISPER
    # =====================================================================================
    async def _extract_audio(self, file_bytes: bytes, ext: str):
        """
        Convert MP3 / WAV audio bytes to text using OpenAI Whisper.
        """

        try:
            # 1. Save bytes to a temporary audio file (Whisper needs a file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name


            # 2. Transcribe
            result = self.whisper_model.transcribe(tmp_path,fp16=False)
            text = result.get("text", "").strip()

            # 3. Remove temp file
            os.remove(tmp_path)

            return {
                "type": "audio",
                "format": ext,
                "text": text
            }

        except Exception as e:
            raise Exception(f"Audio transcription failed: {e}")

    def _extract_images_from_pdf(self, pdf_bytes: bytes) -> List[bytes]:
        images = []
        pdf_document = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)
        return images

    def _extract_text_from_image(self, image_bytes_list: List[bytes]) -> str:
        all_text = []
        for img_bytes in image_bytes_list:
            result = self.reader.readtext(img_bytes, detail=0, paragraph=True)
            all_text.append(" ".join(result))
        return " ".join(all_text)

    def _extract_images_from_docx(self, docx_bytes: bytes) -> List[bytes]:
        images = []
        doc = docx.Document(io.BytesIO(docx_bytes))
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image_part = rel.target_part
                image_bytes = image_part.blob
                images.append(image_bytes)
        return images

    async def _extract_pdf(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        try:
            text = ""
            with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text("text")

            result_text = text.replace("\n", " ").strip()
            if len(text.strip()) < 100:
                images = self._extract_images_from_pdf(file_bytes)
                if images:
                    images_text = self._extract_text_from_image(images)
                    result_text += " " + images_text

            return {"type": "pdf", "text": result_text.strip()}
        except Exception as e:
            raise HTTPException(500, f"PDF extraction failed: {e}")

    async def _extract_image(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(file_bytes))
        with io.BytesIO() as buf:
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        result = self.reader.readtext(img_bytes, detail=0, paragraph=True)
        return {"type": "image", "text": " ".join(result)}

    async def _extract_docx(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        text = docx2txt.process(io.BytesIO(file_bytes))
        try:
            images = self._extract_images_from_docx(file_bytes)
            all_OCR_text = self._extract_text_from_image(images)
            text += all_OCR_text
        except Exception:
            pass  # Ignore if image extraction fails
        return {"type": "word", "text": text}

    async def _extract_excel(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        try:
            with io.BytesIO(file_bytes) as buf:
                engine = (
                    "openpyxl"
                    if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]
                    else "xlrd"
                )
                try:
                    xls = pd.read_excel(buf, sheet_name=None, engine=engine)
                except ImportError:
                    raise HTTPException(
                        500,
                        f"Reading {ext} files requires '{engine}'. Install with 'pip install {engine}>=2.0.1'.",
                    )
            result = {
                sheet: data.to_dict(orient="records") for sheet, data in xls.items()
            }
            return {"type": "excel", "sheets": result}
        except Exception as e:
            raise HTTPException(500, f"Excel extraction failed: {e}")

    async def _extract_json(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        try:
            json_data = json.loads(file_bytes.decode("utf-8"))
            return {"type": "json", "data": json_data}
        except Exception as e:
            raise HTTPException(400, f"Invalid JSON: {e}")

    async def _extract_sql(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        try:
            if ext == ".sql":
                sql_query = (
                    file_bytes.decode("utf-8")
                    .encode("utf-8")
                    .decode("unicode_escape")
                )
                return {"type": "sql_query", "query": sql_query}
            elif ext in [".db", ".sqlite"]:
                return {
                    "type": "sqlite_db",
                    "message": "This is a SQLite database file. Connection is required to extract data.",
                }

        except Exception as e:
            raise HTTPException(400, f"SQL extraction failed: {e}")
        return {}

    async def _extract_zip(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        extracted_data = {}
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            files = zf.namelist()
            for filename in files:
                if filename.endswith("/"):
                    continue
                with zf.open(filename) as inner_file:
                    inner_file_bytes = inner_file.read()
                try:
                    content = inner_file_bytes.decode("utf-8")
                    is_text = True
                except UnicodeDecodeError:
                    content = inner_file_bytes
                    is_text = False

                extracted_data[filename] = {
                    "size_bytes": len(inner_file_bytes),
                    "filename": filename,
                    "is_text": is_text,
                    "content": content,
                }
        return {
            "type": "zip",
            "contained_files": files,
            "file_details": extracted_data,
        }

    async def _extract_rar(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".rar") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        with rarfile.RarFile(tmp_path) as rf:
            files = rf.namelist()
        os.remove(tmp_path)
        return {"type": "rar", "contained_files": files}

async def _extract_html(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
    # Parse HTML
    soup = BeautifulSoup(file_bytes.decode("utf-8", errors="ignore"), "html.parser")

    # Extract text
    text = soup.get_text(separator="\n").strip()

    # Extract links
    links = [a["href"] for a in soup.find_all("a", href=True)]

    # Extract image sources
    img_tags = soup.find_all("img")
    images: List[Dict[str, Any]] = []

    for i, img_tag in enumerate(img_tags):
        src = img_tag.get("src")

        if not src:
            continue

        # Case 1: Base64-encoded inline image
        if src.startswith("data:image"):
            match = re.match(r"data:image/(.*?);base64,(.*)", src)
            if match:
                img_format = match.group(1)
                img_data = base64.b64decode(match.group(2))

                # Convert to PIL image (optional)
                try:
                    image = Image.open(io.BytesIO(img_data))
                    width, height = image.size
                except Exception:
                    width = height = None

                images.append({
                    "index": i,
                    "type": img_format,
                    "width": width,
                    "height": height,
                    "data": img_data,  # raw bytes (you can remove this if not needed)
                })

        # Case 2: Image with external or relative URL
        else:
            images.append({
                "index": i,
                "src": src
            })

    return {
        "type": "html",
        "text": text,
        "links": links,
        "images": images
    }

    async def _extract_txt(self, file_bytes: bytes, ext: str) -> Dict[str, Any]:
        text = file_bytes.decode("utf-8", errors="ignore")
        return {"type": "txt", "text": text}
