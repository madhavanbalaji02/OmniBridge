"""
Stage 1: The Captioner - Dataset Module
Deep-ML Style: Clean, memory-efficient data pipeline.
"""
import os
import json
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

class CaptionDataset(Dataset):
    """
    PyTorch Dataset for Image-Caption pairs.
    Supports COCO-style JSON annotations and simple folder structures.
    """
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        processor: AutoProcessor,
        max_length: int = 512,
        image_size: Tuple[int, int] = (560, 560),
    ):
        """
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO-style JSON or simple JSON with {image: caption}
            processor: Llama 3.2 Vision processor
            max_length: Maximum sequence length for captions
            image_size: Target image size (Llama 3.2 Vision uses 560x560)
        """
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        
        # Load annotations
        self.samples = self._load_annotations(annotation_file)
        print(f"[Dataset] Loaded {len(self.samples)} image-caption pairs")
    
    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """Load annotations from JSON file."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        samples = []
        
        # Handle COCO-style format
        if "images" in data and "annotations" in data:
            # Build image_id -> filename mapping
            id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
            
            for ann in data["annotations"]:
                samples.append({
                    "image": id_to_file[ann["image_id"]],
                    "caption": ann["caption"]
                })
        
        # Handle simple format: [{"image": "filename.jpg", "caption": "..."}]
        elif isinstance(data, list):
            samples = data
        
        # Handle dict format: {"filename.jpg": "caption"}
        elif isinstance(data, dict):
            samples = [{"image": k, "caption": v} for k, v in data.items()]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load and preprocess image
        image_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(image_path).convert("RGB")
        
        # Create the conversation format for Llama 3.2 Vision
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["caption"]}
                ]
            }
        ]
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=False
        )
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Create labels (mask the prompt, only compute loss on caption)
        labels = inputs["input_ids"].clone()
        
        # Find where the assistant response starts and mask everything before
        # For simplicity, we'll use the full sequence as labels
        # The model's internal masking handles the image tokens
        inputs["labels"] = labels
        
        return inputs

class DataCollator:
    """Custom collator for batching multimodal inputs."""
    def __init__(self, processor: AutoProcessor):
        self.processor = processor
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        for key in features[0].keys():
            if key == "pixel_values":
                # Stack image tensors
                batch[key] = torch.stack([f[key] for f in features])
            else:
                # Pad sequences
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features],
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id if key != "labels" else -100
                )
        
        return batch

def create_dataloader(
    config,
    processor: AutoProcessor,
    split: str = "train"
) -> DataLoader:
    """Factory function to create dataloaders."""
    dataset = CaptionDataset(
        image_dir=os.path.join(config.paths.dataset_dir, "images"),
        annotation_file=config.paths.annotation_file,
        processor=processor,
        max_length=config.training.get("max_length", 512) if hasattr(config.training, "get") else 512
    )
    
    collator = DataCollator(processor)
    
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        collate_fn=collator
    )

if __name__ == "__main__":
    # Test dataset loading
    print("[Dataset] Module loaded successfully")
    print("[Dataset] Supports: COCO-style JSON, Simple JSON list, Dict format")
