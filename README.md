# 🌉 OmniBridge - AI Image Captioning

AI-powered image captioning using BLIP-2 vision-language model.

## 🚀 Live Demo

**Try it now:** [https://huggingface.co/spaces/madhavan02/OmniBridge](https://huggingface.co/spaces/madhavan02/OmniBridge)

## Features

- 📷 Upload any image
- 🧠 State-of-the-art BLIP-2 model (Salesforce/blip2-opt-2.7b)
- ⚡ Trained on 400M+ image-text pairs
- 🎯 Accurate, natural language captions

## Tech Stack

- **Model:** Salesforce BLIP-2 (OPT-2.7B)
- **Frontend:** Gradio
- **Hosting:** HuggingFace Spaces
- **Training Infrastructure:** Indiana University Big Red 200 (NVIDIA A100)

## Project Structure

```
OmniBridge/
├── huggingface_space/     # HuggingFace Spaces deployment
│   ├── app.py             # Gradio application
│   ├── requirements.txt   # Dependencies
│   └── README.md          # Space configuration
├── frontend/              # Local demo UI
├── inference.py           # Custom inference code
└── phase1_train.py        # Q-Former training script
```

## Author

**Madhavan Balaji**

---

⭐ Star this repo if you found it helpful!
