from fastapi import APIRouter

from app.api import chat_with_file
from app.api import file_data_extraction
from app.api import CV_parser
from app.api import excel_report
from app.api import youTube_chat
from app.api import voice_conversation
from app.api import image_generation

api_router = APIRouter()

api_router.include_router(
    chat_with_file.router, prefix="/chat", tags=["Chat with File"]
)
api_router.include_router(
    file_data_extraction.router, prefix="/chat", tags=["Chat with File"],
)

api_router.include_router(
    youTube_chat.router,prefix="/ws/chat",tags=["Chat with File"]
)
api_router.include_router(
    voice_conversation.router,prefix="/ws/chat",tags=["Chat with File"]
)
api_router.include_router(CV_parser.router, prefix="/CV_parser", tags=["CV_parser_old"])

api_router.include_router(excel_report.router, prefix="/excel", tags=["Excel Report"])


api_router.include_router(
    image_generation.router, prefix="/image", tags=["Image Generation"]
)
