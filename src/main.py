import os
import sys

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# -------------------------------------------------
# Fix Python path so FastAPI can see src/
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from predict_core import predict_emotion  # ✅ now works

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Emotion Detection App")

# Static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "emotions": None, "text": ""}
    )

@app.post("/", response_class=HTMLResponse)
async def detect_emotion(request: Request, text: str = Form(...)):
    emotions = predict_emotion(text)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "emotions": emotions, "text": text}
    )
