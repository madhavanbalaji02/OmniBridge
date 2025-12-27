#!/usr/bin/env python3
"""
Standalone inference script - runs in subprocess to avoid mutex issues
"""
import sys
import os
import json

# Set threading limits before any imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer, AutoModelForCausalLM, ViTModel

# Q-Former class
class QFormer(nn.Module):
    def __init__(self, image_emb_dim, prompt_len=16, hidden_dim=768):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(prompt_len, image_emb_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=image_emb_dim, num_heads=8)
        self.mlp = nn.Linear(image_emb_dim, hidden_dim)
    
    def forward(self, image_embeds):
        batch_size = image_embeds.size(0)
        query = self.query_tokens.unsqueeze(1).repeat(1, batch_size, 1)
        attn_out, _ = self.cross_attn(query, image_embeds.transpose(0, 1), image_embeds.transpose(0, 1))
        prompt = self.mlp(attn_out).transpose(0, 1)
        return prompt


def generate_caption(image_path, checkpoint_path):
    """Generate caption for a single image"""
    device = torch.device("cpu")
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if '<img>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<img>']})
    img_token_id = tokenizer.convert_tokens_to_ids('<img>')
    
    # Load ViT
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit.eval()
    vit.to(device)
    
    # Load GPT2
    gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
    gpt2.resize_token_embeddings(len(tokenizer))
    
    # Create Q-Former
    q_former = QFormer(
        image_emb_dim=vit.config.hidden_size,
        prompt_len=16,
        hidden_dim=gpt2.config.n_embd
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt2.load_state_dict(checkpoint['gpt2_state'])
    q_former.load_state_dict(checkpoint['qformer'])
    
    gpt2.to(device)
    q_former.to(device)
    gpt2.eval()
    q_former.eval()
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get image features
        image_embeds = vit(image_tensor).last_hidden_state
        prompts = q_former(image_embeds)
        
        # Start generation
        start_text = "A"
        input_ids = tokenizer.encode(start_text, return_tensors='pt').to(device)
        img_token_emb = gpt2.transformer.wte(
            torch.tensor([[img_token_id]], device=device)
        )
        
        generated_tokens = list(input_ids[0].tolist())
        
        for _ in range(50):
            gpt2_inputs = gpt2.transformer.wte(input_ids)
            gpt2_inputs = torch.cat([img_token_emb, prompts, gpt2_inputs], dim=1)
            
            outputs = gpt2(inputs_embeds=gpt2_inputs)
            logits = outputs.logits[0, -1, :] / 0.8
            
            # Top-k
            top_k = 50
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            decoded = tokenizer.decode([next_token.item()])
            if '.' in decoded and len(generated_tokens) > 10:
                break
        
        caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return caption.strip()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: caption_worker.py <image_path> <checkpoint_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    
    try:
        caption = generate_caption(image_path, checkpoint_path)
        print(json.dumps({"success": True, "caption": caption}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)
