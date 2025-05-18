from fastapi import FastAPI
from app.api.routes import detect, websocket
from app.core.config import add_cors
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS
add_cors(app)

# Include API routes
app.include_router(detect.router, prefix="/api", tags=["Detection"])
app.include_router(websocket.router, prefix="/ws", tags=["Detection"])
