"""
FastAPI Backend for OmniBridge Image Captioning
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
from inference import CaptionGenerator

app = FastAPI(title="OmniBridge Caption Generator", version="1.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model (loaded once)
model = None

# Checkpoint path - adjust based on your setup
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", 
    "./checkpoints/checkpoint_epoch20.pth"
)


def get_model():
    global model
    if model is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise HTTPException(
                status_code=500, 
                detail=f"Checkpoint not found at {CHECKPOINT_PATH}. Please download from Big Red."
            )
        model = CaptionGenerator(CHECKPOINT_PATH, device="cpu")
    return model


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page"""
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/generate")
async def generate_caption(file: UploadFile = File(...)):
    """Generate caption for uploaded image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Generate caption
        generator = get_model()
        caption = generator.generate_from_pil(image)
        
        return {
            "success": True,
            "caption": caption,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "architecture": "Q-Former + ViT + DistilGPT2",
        "vision_encoder": "google/vit-base-patch16-224-in21k",
        "language_model": "distilgpt2",
        "trainable_params": "~70M",
        "training_epochs": 20,
        "final_loss": 0.1667
    }


# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
