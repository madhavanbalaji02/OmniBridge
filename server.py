"""
Flask Backend for OmniBridge Image Captioning
Simple HTTP server for caption generation
"""
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import os
import torch
from inference import CaptionGenerator

app = Flask(__name__, static_folder='frontend')

# Global model (loaded once)
model = None

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", 
    "./checkpoints/checkpoint_epoch20.pth"
)


def get_model():
    global model
    if model is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise Exception(f"Checkpoint not found at {CHECKPOINT_PATH}")
        print("Loading model... (this may take a minute)")
        model = CaptionGenerator(CHECKPOINT_PATH, device="cpu")
    return model


@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory('frontend', 'index.html')


@app.route('/health')
def health():
    """Health check"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route('/generate', methods=['POST'])
def generate():
    """Generate caption for uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Read image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Generate caption
        generator = get_model()
        caption = generator.generate_from_pil(image)
        
        return jsonify({
            "success": True,
            "caption": caption,
            "filename": file.filename
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/model-info')
def model_info():
    """Get model information"""
    return jsonify({
        "architecture": "Q-Former + ViT + DistilGPT2",
        "vision_encoder": "google/vit-base-patch16-224-in21k",
        "language_model": "distilgpt2",
        "trainable_params": "~70M",
        "training_epochs": 20,
        "final_loss": 0.1667
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  OmniBridge Caption Generator")
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
