"""
BLIP-2 Fine-tuning on TextCaps Dataset using LoRA
Optimized for Big Red 200 A100 GPU

This script fine-tunes BLIP-2 to generate captions that include text visible in images.
Uses LoRA (Low-Rank Adaptation) for efficient training.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import json
import logging
from datetime import datetime

# Set cache directory to scratch (avoid home quota issues)
import os
os.environ['HF_HOME'] = '/N/scratch/madbala/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/N/scratch/madbala/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/N/scratch/madbala/.cache/huggingface/hub'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# ============== Configuration ==============
class Config:
    # Model
    model_name = "Salesforce/blip2-opt-2.7b"
    
    # Training
    batch_size = 4
    gradient_accumulation_steps = 4  # Effective batch = 16
    learning_rate = 2e-4
    num_epochs = 3
    warmup_ratio = 0.1
    max_length = 64
    
    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    
    # Data
    max_train_samples = 20000  # Use subset for faster training
    max_val_samples = 1000
    
    # Paths
    output_dir = "./blip2_textcaps_lora"
    checkpoint_dir = "./checkpoints_textcaps"


# ============== Dataset ==============
class TextCapsDataset(Dataset):
    """TextCaps dataset for image captioning with text understanding"""
    
    def __init__(self, processor, split="train", max_samples=None):
        logger.info(f"Loading TextCaps {split} dataset...")
        self.processor = processor
        
        # Load TextCaps from HuggingFace
        dataset = load_dataset("lmms-lab/TextCaps", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = dataset
        logger.info(f"Loaded {len(self.data)} samples for {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Get image
            image = item['image']
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert('RGB')
            
            # Get caption (use first reference caption)
            if isinstance(item['reference_strs'], list):
                caption = item['reference_strs'][0]
            else:
                caption = item['reference_strs']
            
            # Process inputs
            encoding = self.processor(
                images=image,
                text=caption,
                padding="max_length",
                truncation=True,
                max_length=Config.max_length,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            
            # Create labels (same as input_ids for causal LM)
            encoding['labels'] = encoding['input_ids'].clone()
            
            return encoding
            
        except Exception as e:
            logger.warning(f"Error processing item {idx}: {e}")
            # Return dummy data
            dummy = self.processor(
                images=Image.new('RGB', (224, 224)),
                text="placeholder",
                padding="max_length",
                max_length=Config.max_length,
                return_tensors="pt"
            )
            dummy = {k: v.squeeze(0) for k, v in dummy.items()}
            dummy['labels'] = dummy['input_ids'].clone()
            return dummy


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'pixel_values':
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated


# ============== Model Setup ==============
def setup_model():
    """Load BLIP-2 and apply LoRA"""
    logger.info(f"Loading {Config.model_name}...")
    
    # Load processor
    processor = Blip2Processor.from_pretrained(Config.model_name)
    
    # Load model in FP16
    model = Blip2ForConditionalGeneration.from_pretrained(
        Config.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, processor


# ============== Training ==============
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for step, batch in enumerate(progress_bar):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / Config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * Config.gradient_accumulation_steps
        progress_bar.set_postfix({
            'loss': f"{loss.item() * Config.gradient_accumulation_steps:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


def generate_sample_captions(model, processor, device, num_samples=5):
    """Generate sample captions to show progress"""
    model.eval()
    
    # Load a few test images
    test_dataset = load_dataset("lmms-lab/TextCaps", split="test")
    
    logger.info("\n" + "="*50)
    logger.info("Sample Generated Captions:")
    logger.info("="*50)
    
    for i in range(min(num_samples, len(test_dataset))):
        item = test_dataset[i]
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ground_truth = item['reference_strs'][0] if isinstance(item['reference_strs'], list) else item['reference_strs']
        
        logger.info(f"\nImage {i+1}:")
        logger.info(f"  Generated: {caption.strip()}")
        logger.info(f"  Ground Truth: {ground_truth}")
    
    logger.info("="*50 + "\n")


def main():
    """Main training function"""
    logger.info("="*60)
    logger.info("BLIP-2 Fine-tuning on TextCaps with LoRA")
    logger.info("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create directories
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    # Setup model
    model, processor = setup_model()
    
    # Create datasets
    train_dataset = TextCapsDataset(processor, split="train", max_samples=Config.max_train_samples)
    val_dataset = TextCapsDataset(processor, split="val", max_samples=Config.max_val_samples)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=0.01
    )
    
    # Scheduler
    total_steps = len(train_loader) * Config.num_epochs // Config.gradient_accumulation_steps
    warmup_steps = int(total_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Total training samples: {len(train_dataset)}")
    logger.info(f"  Batch size: {Config.batch_size} x {Config.gradient_accumulation_steps} = {Config.batch_size * Config.gradient_accumulation_steps}")
    logger.info(f"  Epochs: {Config.num_epochs}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(Config.num_epochs):
        logger.info(f"\n{'='*20} Epoch {epoch+1}/{Config.num_epochs} {'='*20}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, Config.num_epochs)
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
        
        # Generate sample captions
        generate_sample_captions(model, processor, device)
        
        # Save checkpoint
        checkpoint_path = os.path.join(Config.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        model.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(Config.output_dir, "best_model")
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            logger.info(f"New best model saved: {best_path}")
    
    # Save final model
    final_path = os.path.join(Config.output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Final evaluation
    logger.info("\nFinal Sample Captions:")
    generate_sample_captions(model, processor, device, num_samples=10)


if __name__ == "__main__":
    main()
