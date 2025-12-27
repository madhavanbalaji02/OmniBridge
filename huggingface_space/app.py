import gradio as gr
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load model (happens once when Space starts)
print("Loading BLIP-2 model...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
if torch.cuda.is_available():
    model = model.to("cuda")
model.eval()
print("Model loaded!")


def generate_caption(image):
    """Generate caption for uploaded image"""
    if image is None:
        return "Please upload an image"
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda", torch.float16) for k, v in inputs.items()}
    
    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    
    caption = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    return caption


# Create Gradio interface
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption", lines=3),
    title="🌉 OmniBridge - AI Image Captioning",
    description="""
    <div style="text-align: center;">
        <p>Upload an image to generate an AI caption using <strong>BLIP-2</strong></p>
        <p>Powered by Salesforce/blip2-opt-2.7b (trained on 400M+ images)</p>
    </div>
    """,
    examples=[
        ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"],
    ],
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { max-width: 800px !important; }
        footer { display: none !important; }
    """
)

if __name__ == "__main__":
    demo.launch()
