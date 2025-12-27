# 🌉 OmniBridge - AI Image Captioning

AI-powered image captioning using BLIP-2 vision-language model, including a custom fine-tuned version that reads text in images!

## 🚀 Live Demos

| Model | Link | Description |
|-------|------|-------------|
| **Pretrained BLIP-2** | [🔗 Demo](https://huggingface.co/spaces/madhavan02/OmniBridge) | Base model trained on 400M+ images |
| **Fine-tuned on TextCaps** | [🔗 Demo](https://huggingface.co/spaces/madhavan02/OmniBridge-TextCaps) | Custom LoRA fine-tuned to read text in images! ✨ |

## ✨ What's Special About the Fine-tuned Model?

The fine-tuned model was trained using **LoRA (Low-Rank Adaptation)** on the TextCaps dataset to recognize and include visible text in captions.

| Input | Pretrained | Fine-tuned |
|-------|------------|------------|
| Sign with text | "A sign on a building" | "A sign on a building that says 'OPEN'" |
| Book cover | "A stack of books" | "A book titled 'The Art of War'" |
| Jersey | "A person in a shirt" | "A player wearing an Adidas jersey" |

## Features

- 📷 Upload any image for instant captioning
- 🧠 State-of-the-art BLIP-2 model (Salesforce/blip2-opt-2.7b)
- ⚡ Pretrained on 400M+ image-text pairs
- 📝 Fine-tuned on 20K TextCaps images for text recognition
- 🎯 Accurate, natural language captions

## Tech Stack

- **Base Model:** Salesforce BLIP-2 (OPT-2.7B) - 3.8B parameters
- **Fine-tuning:** LoRA (Low-Rank Adaptation) - 33M trainable parameters
- **Dataset:** TextCaps (images containing readable text)
- **Training Time:** ~1 hour on NVIDIA A100-40GB
- **Frontend:** Gradio
- **Hosting:** HuggingFace Spaces
- **Training Infrastructure:** Indiana University Big Red 200

## Project Structure

```
OmniBridge/
├── huggingface_space/       # Pretrained BLIP-2 deployment
│   ├── app.py               # Gradio application
│   └── requirements.txt     # Dependencies
├── lora_weights/            # Fine-tuned LoRA weights
│   ├── adapter_model.safetensors  # Trained weights (126MB)
│   └── adapter_config.json
├── finetune_blip2.py        # LoRA fine-tuning script
├── run_finetune.slurm       # Slurm job script for HPC
├── frontend/                # Local demo UI
└── phase1_train.py          # Q-Former training script
```

## Fine-tuning Details

The model was fine-tuned using LoRA with these hyperparameters:

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 2e-4 |
| Batch Size | 16 (4 × 4 accumulation) |
| Epochs | 3 |
| Training Samples | 20,000 |
| Final Loss | 0.59 |

## Running Locally

```bash
# Clone the repo
git clone https://github.com/madhavanbalaji02/OmniBridge.git
cd OmniBridge

# Install dependencies
pip install torch transformers peft gradio

# Run the fine-tuned demo
cd huggingface_space && python app.py
```

## Author

**Madhavan Balaji**  
Indiana University Bloomington

---

⭐ Star this repo if you found it helpful!
