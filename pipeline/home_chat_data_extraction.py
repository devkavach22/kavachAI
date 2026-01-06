import os
import io
import json
import zipfile
import rarfile
import tempfile
from typing import Dict, Any, List
import re
import os
import base64
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import docx
import docx2txt
import pandas as pd
from bs4 import BeautifulSoup
import cv2  # OpenCV for video frame extraction
from faster_whisper import WhisperModel  # 🚀 Faster Whisper

class FileDataExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(["en"])
        
        # 🚀 Use faster-whisper (5-10x faster than openai-whisper)
        self.whisper_model = WhisperModel("base", device=os.getenv("FASTER_WHISPER_DEVICE","cpu"), compute_type=os.getenv("FASTER_WHISPER_COMPUTE_TYPE","int8"))

    async def extract_data(self, file_bytes: bytes, ext: str, content_type: str = None) -> str:
        """
        Extract TEXT only from file bytes given extension and content type.
        
        Args:
            file_bytes: Raw file content as bytes
            ext: File extension (e.g., '.pdf', '.mp4')
            content_type: MIME type (optional, for additional validation)
            
        Returns:
            str: Extracted text content
        """
        ext = ext.lower()
        
        extraction_methods = {
            ".pdf": self._extract_pdf_text,
            ".jpg": self._extract_image_text,
            ".jpeg": self._extract_image_text,
            ".png": self._extract_image_text,
            ".bmp": self._extract_image_text,
            ".tiff": self._extract_image_text,
            ".docx": self._extract_docx_text,
            ".xls": self._extract_excel_text,
            ".xlsx": self._extract_excel_text,
            ".xlsm": self._extract_excel_text,
            ".xltx": self._extract_excel_text,
            ".xltm": self._extract_excel_text,
            ".json": self._extract_json_text,
            ".db": self._extract_sql_text,
            ".sqlite": self._extract_sql_text,
            ".sql": self._extract_sql_text,
            ".zip": self._extract_zip_text,
            ".rar": self._extract_rar_text,
            ".html": self._extract_html_text,
            ".txt": self._extract_txt_text,
            ".mp3": self._extract_audio_text,
            ".wav": self._extract_audio_text,
            ".mp4": self._extract_video_text,
            ".avi": self._extract_video_text,
            ".mov": self._extract_video_text,
            ".mkv": self._extract_video_text,
            ".webm": self._extract_video_text,
            ".flv": self._extract_video_text,
            ".wmv": self._extract_video_text,
        }

        if ext in extraction_methods:
            try:
                return await extraction_methods[ext](file_bytes)
            except Exception as e:
                return f"Extraction failed for {ext}: {str(e)}"
        else:
            return f"Unsupported file type: {ext}"

    # =====================================================================================
    # VIDEO → TEXT ONLY (FASTER WHISPER + FRAME OCR)
    # =====================================================================================
    async def _extract_video_text(self, file_bytes: bytes) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(file_bytes)
                video_path = tmp.name

            # 🚀 Faster Whisper audio transcription
            audio_text = ""
            try:
                segments, _ = self.whisper_model.transcribe(video_path, beam_size=5)
                audio_text = " ".join(segment.text.strip() for segment in segments).strip()
            except:
                pass

            # Frame OCR
            frame_text = await self._extract_text_from_video_frames(video_path)

            os.remove(video_path)
            return f"{audio_text} {frame_text}".strip()
        except Exception as e:
            return f"Video extraction failed: {e}"

    # =====================================================================================
    # AUDIO → TEXT ONLY (FASTER WHISPER)
    # =====================================================================================
    async def _extract_audio_text(self, file_bytes: bytes) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # 🚀 Faster Whisper transcription (5-10x faster)
            segments, _ = self.whisper_model.transcribe(tmp_path, beam_size=5)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            
            os.remove(tmp_path)
            return text
        except:
            return "Audio transcription failed"

    async def _extract_text_from_video_frames(self, video_path: str) -> str:
        try:
            cap = cv2.VideoCapture(video_path)
            frame_texts = []
            frame_count = 0
            keyframe_interval = 30

            while cap.isOpened() and frame_count < 150:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % keyframe_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    with io.BytesIO() as buf:
                        pil_image.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    
                    result = self.reader.readtext(img_bytes, detail=0, paragraph=True)
                    frame_texts.append(" ".join(result))

                frame_count += 1

            cap.release()
            return " ".join(frame_texts)
        except:
            return ""

    # =====================================================================================
    # PDF → TEXT ONLY
    # =====================================================================================
    async def _extract_pdf_text(self, file_bytes: bytes) -> str:
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
            return result_text.strip()
        except:
            return "PDF extraction failed"

    # =====================================================================================
    # IMAGE → TEXT ONLY
    # =====================================================================================
    async def _extract_image_text(self, file_bytes: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            with io.BytesIO() as buf:
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            result = self.reader.readtext(img_bytes, detail=0, paragraph=True)
            return " ".join(result)
        except:
            return "Image OCR failed"

    # =====================================================================================
    # DOCX → TEXT ONLY
    # =====================================================================================
    async def _extract_docx_text(self, file_bytes: bytes) -> str:
        try:
            text = docx2txt.process(io.BytesIO(file_bytes))
            try:
                images = self._extract_images_from_docx(file_bytes)
                all_OCR_text = self._extract_text_from_image(images)
                text += all_OCR_text
            except:
                pass
            return text.strip()
        except:
            return "DOCX extraction failed"

    # =====================================================================================
    # EXCEL → TEXT ONLY
    # =====================================================================================
    async def _extract_excel_text(self, file_bytes: bytes) -> str:
        try:
            with io.BytesIO(file_bytes) as buf:
                xls = pd.read_excel(buf, sheet_name=None)
            all_text = []
            for sheet_name, data in xls.items():
                all_text.append(f"SHEET: {sheet_name}\n{data.to_string()}")
            return "\n\n".join(all_text)
        except:
            return "Excel extraction failed"

    # =====================================================================================
    # JSON → TEXT ONLY
    # =====================================================================================
    async def _extract_json_text(self, file_bytes: bytes) -> str:
        try:
            json_data = json.loads(file_bytes.decode("utf-8"))
            return json.dumps(json_data, indent=2)
        except:
            return "Invalid JSON"

    # =====================================================================================
    # SQL → TEXT ONLY
    # =====================================================================================
    async def _extract_sql_text(self, file_bytes: bytes) -> str:
        try:
            if file_bytes.decode("utf-8", errors="ignore").strip().startswith("--") or ".db" in str(file_bytes)[:100]:
                return "SQLite database or SQL file detected. Database connection required for data extraction."
            return file_bytes.decode("utf-8", errors="ignore").strip()
        except:
            return "SQL extraction failed"

    # =====================================================================================
    # ZIP → TEXT ONLY
    # =====================================================================================
    async def _extract_zip_text(self, file_bytes: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                files = zf.namelist()
                text_content = []
                for filename in files:
                    if filename.endswith("/"):
                        continue
                    with zf.open(filename) as inner_file:
                        inner_bytes = inner_file.read()
                        try:
                            text_content.append(f"FILE: {filename}\n{inner_bytes.decode('utf-8')}")
                        except:
                            text_content.append(f"FILE: {filename} (binary, {len(inner_bytes)} bytes)")
                return "\n\n".join(text_content)
        except:
            return "ZIP extraction failed"

    # =====================================================================================
    # RAR → TEXT ONLY (list only)
    # =====================================================================================
    async def _extract_rar_text(self, file_bytes: bytes) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.rar') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            with rarfile.RarFile(tmp_path) as rf:
                files = rf.namelist()
            os.remove(tmp_path)
            return f"RAR contains {len(files)} files:\n" + "\n".join(files)
        except:
            return "RAR extraction failed"

    # =====================================================================================
    # HTML → TEXT ONLY
    # =====================================================================================
    async def _extract_html_text(self, file_bytes: bytes) -> str:
        try:
            soup = BeautifulSoup(file_bytes.decode("utf-8", errors="ignore"), "html.parser")
            return soup.get_text(separator=" ").strip()
        except:
            return "HTML extraction failed"

    # =====================================================================================
    # TXT → TEXT ONLY
    # =====================================================================================
    async def _extract_txt_text(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="ignore").strip()

    # HELPER METHODS (unchanged)
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
        pdf_document.close()
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
