# Project

**Project ** is a powerful API designed for extracting structured data from CVs (Resumes) and enabling interactive chat with document contents using AI. It leverages advanced OCR, semantic search, and Large Language Models (LLMs) to provide deep insights into candidate profiles.

## 🚀 Features

- **📄 CV Parsing**: Automatically extract text and structured data from PDF resumes.
- **💬 Chat with PDF**: Upload a document and chat with it to ask specific questions.
- **🧠 Semantic Search**: Uses Vector Database (ChromaDB) for efficient context retrieval.
- **⚡ Real-time Chat**: WebSocket support for instant, interactive query processing.
- **🔍 OCR Support**: Integrated with PyMuPDF and PyTesseract to handle image-based PDFs.

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Language**: Python 3.10+
- **Database**: [ChromaDB](https://www.trychroma.com/) (Vector Store)
- **AI/ML**: LangChain, PyMuPDF (Fitz), PyTesseract, PDF2Image
- **Package Manager**: `uv` (or `pip`)

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd Project
    ```

2.  **Set up the environment**
    Create a `.env` file in the root directory and add your necessary API keys (e.g., OpenAI, Database URLs).
    ```bash
    cp .env.example .env
    ```

3.  **Install dependencies**
    Using `uv` (recommended):
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### Run the Development Server

Start the FastAPI server with hot-reloading:

```bash
uv run fastapi dev main.py
```

The API will be available at `http://localhost:8000`.

### API Documentation

Interactive API docs are automatically generated:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### WebSocket Chat Demo

You can test the real-time chat functionality by visiting:
`http://localhost:8000/ws-chat`

## 📂 Project Structure

```
├── app/
│   ├── api/            # API endpoints (Chat, CV Parser)
│   ├── router.py       # Main API router
│   └── ...
├── db/                 # Database connection logic
├── services/           # Business logic (Extraction, Chains)
├── RAG/                # RAG (Retrieval-Augmented Generation) components
├── main.py             # Application entry point
└── README.md           # Project documentation
```

### Conversation Flow

2️⃣ Correct Architecture (React ↔ Pipecat)
'''
┌────────────┐        WebSocket        ┌─────────────────────┐
│ React App  │  <-------------------> │ FastAPI + Pipecat    │
│ (Browser)  │   Audio + Frames       │ (Python backend)     │
└────────────┘                         └─────────────────────┘
       🎤                                  🧠 STT → LLM → TTS
       🔊                                   🔊
'''

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
