import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv

def add_cors(app: FastAPI):
    load_dotenv()
    origins_env = os.getenv("ALLOWED_ORIGINS", "*")
    
    # Convert comma-separated string to list, or fallback to ["*"]
    if origins_env == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
