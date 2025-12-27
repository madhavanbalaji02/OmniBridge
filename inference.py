"""
Inference Script for Q-Former Image Captioning
Loads trained model and generates captions for input images
"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer, AutoModelForCausalLM, ViTModel
import argparse
import os

# ============ Q-FORMER (same as training) ============
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


class CaptionGenerator:
    def __init__(self, checkpoint_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if '<img>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<img>']})
        self.img_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        
        # Load ViT (frozen)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit.eval()
        self.vit.to(self.device)
        
        # Load GPT2
        self.gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.gpt2.resize_token_embeddings(len(self.tokenizer))
        
        # Create Q-Former
        self.q_former = QFormer(
            image_emb_dim=self.vit.config.hidden_size,
            prompt_len=16,
            hidden_dim=self.gpt2.config.n_embd
        )
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gpt2.load_state_dict(checkpoint['gpt2_state'])
        self.q_former.load_state_dict(checkpoint['qformer'])
        
        self.gpt2.to(self.device)
        self.q_former.to(self.device)
        
        self.gpt2.eval()
        self.q_former.eval()
        
        print("Model loaded successfully!")
    
    def generate_caption(self, image_path: str, max_length: int = 30) -> str:
        """Generate caption for an image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get image embeddings from ViT
            image_embeds = self.vit(image_tensor).last_hidden_state
            
            # Project through Q-Former
            prompts = self.q_former(image_embeds)
            
            # Prepare initial input
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
            img_token_emb = self.gpt2.transformer.wte(
                torch.tensor([[self.img_token_id]], device=self.device)
            )
            
            generated_tokens = []
            
            # Autoregressive generation
            for _ in range(max_length):
                gpt2_inputs = self.gpt2.transformer.wte(input_ids)
                gpt2_inputs = torch.cat([img_token_emb, prompts, gpt2_inputs], dim=1)
                
                outputs = self.gpt2(inputs_embeds=gpt2_inputs)
                logits = outputs.logits[0, -1, :]
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            caption = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return caption.strip()
    
    def generate_from_pil(self, pil_image: Image.Image, max_length: int = 50) -> str:
        """Generate caption from PIL Image object"""
        image_tensor = self.transform(pil_image.convert('RGB')).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_embeds = self.vit(image_tensor).last_hidden_state
            prompts = self.q_former(image_embeds)
            
            # Start with a space token instead of BOS (GPT2 has no BOS)
            start_text = "A"
            input_ids = self.tokenizer.encode(start_text, return_tensors='pt').to(self.device)
            
            img_token_emb = self.gpt2.transformer.wte(
                torch.tensor([[self.img_token_id]], device=self.device)
            )
            
            generated_tokens = list(input_ids[0].tolist())
            
            for _ in range(max_length):
                gpt2_inputs = self.gpt2.transformer.wte(input_ids)
                gpt2_inputs = torch.cat([img_token_emb, prompts, gpt2_inputs], dim=1)
                
                outputs = self.gpt2(inputs_embeds=gpt2_inputs)
                logits = outputs.logits[0, -1, :]
                
                # Apply temperature
                logits = logits / 0.8
                
                # Top-k filtering
                top_k = 50
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop on EOS or period followed by space-like tokens
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit a sentence ending
                decoded = self.tokenizer.decode([next_token.item()])
                if '.' in decoded and len(generated_tokens) > 10:
                    break
            
            caption = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return caption.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    generator = CaptionGenerator(args.checkpoint, args.device)
    caption = generator.generate_caption(args.image)
    print(f"\nGenerated Caption: {caption}")
