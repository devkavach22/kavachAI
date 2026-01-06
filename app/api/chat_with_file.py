from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from db.connection import CV_data_collection
from RAG.data_processing.chunker import chunk_pdf_text
import random
from services.data_extraction import FileDataExtractor
from services.chat_with_file_chain import run_qa_query

from typing import Optional

router = APIRouter()


@router.post("/upload_and_chat/")
async def upload_and_chat(file: UploadFile = File(...)):
    """
    Extract text and image data (with OCR) from a PDF using PyMuPDF4LLM.
    """
    try:
        # Read uploaded content into memory (we process in-memory; no temp file)
        pdf_content = await file.read()

        # --- Step 1: Extract text with  ---
        extractor = FileDataExtractor()
        pdf_data = await extractor._extract_pdf(pdf_content, "pdf")

        # text data for chunking
        pdf_text_data = pdf_data["text"]

        # chunks the text data and ocr text
        # List_of_chunks = chunk_pdf_text(text_data)

        # and  List_of_chunks tp the   collection
        # and 10^10 - 1 (largest 10-digit number).
        chat_ID = random.randint(10**9, 10**10 - 1)
        CV_data_collection.add(
            ids=[f"id_{chat_ID}"],
            documents=[pdf_text_data],
            metadatas=[{"chat_ID": chat_ID}],
        )
        # similar_docs = CV_data_collection.similarity_search("over 3.4 years of experience ", k=2,filter={"chat_ID": chat_ID})

        #  Get all items (IDs, documents, and metadata)
        # The get() method without any arguments retrieves everything.
        collection_data = CV_data_collection.get(include=["documents", "metadatas"])
        # CV_data_collection.

        return JSONResponse(
            content={
                "chat_ID": chat_ID,
                "file status": "file processed successfully",
                # "collection_data": collection_data,
            },
            status_code=200,
        )

    except Exception as e:
        print("Error during processing:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        try:
            pass
        except Exception as e:
            print(f"Error during cleanup: {e}")


# ✅ Helper: search + store
@router.post("/chat_with_file/{chat_ID}")
async def chat_with_file(chat_ID: int, query: str = ""):
    if not chat_ID:
        raise HTTPException(status_code=400, detail="chat ID is required")
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")

    existing_docs = CV_data_collection.get(where={"chat_ID": chat_ID})
    if existing_docs["ids"]:
        pass
    else:
        return JSONResponse(
            content={
                "message": f"Chat ID is Invalid ID {chat_ID}",
            }
        )

    try:
        # ✅ Perform semantic search limited to this chat_ID
        results = CV_data_collection.query(
            query_texts=[query],
            n_results=2,
            where={"chat_ID": chat_ID},  # Filter by metadata
        )

        # ✅ Prepare readable response
        docs = []
        for i in range(len(results.get("ids", [[]])[0])):
            docs.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )


        #  Calling the Cheain
        answer = run_qa_query(query, docs[0]["document"])

        return JSONResponse(
            content={
                "message": f"Chat query processed for ID {chat_ID}",
                "answer": answer,
                "results": docs,
            }
        )

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat_with_file/ws")
async def get_ws_upload_and_chat():
    """
    WebSocket endpoint for chat.

    Connect specific path: `/api/chat/chat_with_file/ws`

    **Usage:**
    - Connect via WebSocket client.
    - Send JSON: `{"chat_ID": "123", "query": "hello"}`
    - Receive JSON: `{"ai_answer": "..."}`
    """
    return JSONResponse(
        content={
            "message": "This is a WebSocket endpoint. Please use a WebSocket client to connect."
        },
        status_code=426,
    )


@router.websocket("/chat_with_file/ws")
async def ws_upload_and_chat(websocket: WebSocket):
    # Accept WebSocket connection
    await websocket.accept()
    # print("🔵 Client connected:", websocket.client)

    # Initial check for chat_ID validity
    # if chat_ID: # Only check if chat_ID is provided in the path
    #     existing_docs = CV_data_collection.get(where={"chat_ID": chat_ID})
    #     if not existing_docs["ids"]:
    #         await websocket.send_json({"error": f"Chat ID is Invalid ID {chat_ID}"})
    #         await websocket.close()
    #         return

    try:
        while True:
            # 🟢 Receive JSON message
            data = await websocket.receive_json()

            chat_ID = str(data.get("chat_ID", "")).strip()
            query = str(data.get("query", "")).strip()

            # Validate Chat ID
            if not chat_ID.isdigit():
                await websocket.send_json({"error": "chat_ID must be a number"})
                continue

            # Validate Query
            if not query:
                await websocket.send_json({"error": "Query text is required"})
                continue

            # Validate if chat_ID exists in DB
            existing_docs = CV_data_collection.get(where={"chat_ID": int(chat_ID)})
            if not existing_docs["ids"]:
                await websocket.send_json({"error": f"Chat ID is Invalid: {chat_ID}"})
                continue

            # ================================
            # 🔍 SEMANTIC SEARCH
            # ================================
            results = CV_data_collection.query(
                query_texts=[query], n_results=2, where={"chat_ID": int(chat_ID)}
            )

            docs = []
            for i in range(len(results.get("ids", [[]])[0])):
                docs.append(
                    {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                    }
                )

            # 🧠 QA Chain
            ai_answer = run_qa_query(query, docs[0]["document"] if docs else "")

            # Send back to client
            await websocket.send_json(
                {
                    "message": f"Chat processed for ID {chat_ID}",
                    "ai_answer": ai_answer,
                    # "results": docs,
                }
            )

    except WebSocketDisconnect:
        # Silently handle disconnects to avoid console overhead in production
        pass

    except Exception as e:
        import logging
        # Use logging instead of print for better performance and log management
        logging.error(f"WebSocket Error: {e}")
        try:
            await websocket.send_json({"error": "Internal server error"})
        except Exception:
            pass

    finally:
        # Final disconnect cleanup
        await websocket.close()
        print(f"⚪ Connection closed for: {websocket.client}")
