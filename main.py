from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router import api_router
from fastapi.responses import HTMLResponse
from scalar_fastapi import get_scalar_api_reference
from db import excelDB
from contextlib import asynccontextmanager

# Loading environment variables from .env
load_dotenv()



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: make a database connection at the start of the application
    excelDB.connect_to_db()
    yield
    # Shutdown: cleanup code can go here if needed
    pass


app = FastAPI(
    title="Project", description="API for extracting CV data", lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Include the API router
# app.include_router(cv_router, prefix="/api", tags=["CV Data Extract"])
app.include_router(api_router, prefix="/api")

# app.include_router(api_router, prefix="/ws")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


html = """
<!DOCTYPE html>
<html>
<head>
    <title>YouTube WebSocket Chat</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen flex justify-center items-center p-6">

<div class="w-full max-w-2xl bg-gray-800 rounded-2xl shadow-xl p-6 space-y-6">

    <h1 class="text-3xl font-bold text-center">📡 YouTube Transcript Chat</h1>
    <p class="text-center text-gray-300">Ask anything about a YouTube video in real-time!</p>

    <div class="p-3 rounded-lg text-center font-semibold" id="status" class="bg-gray-700">
        Connecting...
    </div>

    <!-- STEP 1 — Enter YouTube URL -->
    <form id="linkForm" class="space-y-3">
        <label class="font-semibold">Paste YouTube URL</label>
        <input id="youtubeLink" type="text" placeholder="https://youtube.com/watch?v=..."
               class="w-full px-3 py-2 rounded-lg bg-gray-700 border border-gray-600 focus:ring focus:ring-blue-500 outline-none" required>
        <button class="w-full py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition">
            Load Transcript
        </button>
    </form>

    <!-- STEP 2 — Query Text -->
    <form id="queryForm" style="display:none;" class="space-y-3">
        <label class="font-semibold">Ask a Question</label>
        <input id="queryText" type="text" placeholder="e.g. What is the video about?"
               class="w-full px-3 py-2 rounded-lg bg-gray-700 border border-gray-600 focus:ring focus:ring-green-500 outline-none" required>
        <button class="w-full py-2 bg-green-600 hover:bg-green-700 rounded-lg font-semibold transition">
            Send Message
        </button>
    </form>

    <div class="border-t border-gray-600 pt-4">
        <h2 class="font-bold text-lg">📩 Chat Responses</h2>
        <ul id="messages" class="space-y-2 max-h-64 overflow-y-auto p-2 bg-gray-700 rounded-lg text-sm"></ul>
    </div>

</div>

<script>
        let ws = new WebSocket("ws://localhost:8000/api/ws/chat/youtube_chat");
        let transcriptReady = false;

        ws.onopen = () => {
            document.getElementById("status").textContent = "🟢 Connected to server";
            document.getElementById("status").className = "bg-green-600 p-3 rounded-lg text-center";
        };

        ws.onerror = () => {
            document.getElementById("status").textContent = "🔴 Connection Error";
            document.getElementById("status").className = "bg-red-600 p-3 rounded-lg text-center";
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            const li = document.createElement("li");

            li.className = "bg-gray-600 p-2 rounded-lg border border-gray-500";

            li.textContent = JSON.stringify(msg, null, 2);
            document.getElementById("messages").appendChild(li);

            if (msg.status === "transcript_loaded") {
                transcriptReady = true;
                document.getElementById("queryForm").style.display = "block";
                alert("Transcript Loaded ✔ Now ask questions!");
            }
        };

        document.getElementById("linkForm").onsubmit = (e) => {
            e.preventDefault();
            const link = document.getElementById("youtubeLink").value.trim();
            ws.send(JSON.stringify({ youtube_link: link }));
            document.getElementById("status").textContent = "💬 Loading transcript...";
            document.getElementById("status").className = "bg-yellow-500 p-3 rounded-lg text-black";
        };

        document.getElementById("queryForm").onsubmit = (e) => {
            e.preventDefault();
            if (!transcriptReady) return alert("Transcript not ready yet!");

            const query = document.getElementById("queryText").value.trim();
            ws.send(JSON.stringify({ query }));

            document.getElementById("queryText").value = "";
        };
</script>

</body>
</html>
"""


@app.get("/ws-chat")
async def get():
    return HTMLResponse(html)

