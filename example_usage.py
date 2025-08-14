#!/usr/bin/env python3
"""
Example usage of the trained FlexTok Encoder

This script demonstrates how to load and use a trained FlexTok encoder
for encoding images into tokens.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add the flextok package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'flextok'))

try:
    from flextok.model.trunks.transformers import FlexTransformer
    from flextok.model.preprocessors.patching import ImagePatcher
    from flextok.model.preprocessors.linear import LinearProjector
except ImportError as e:
    print(f"Error importing flextok: {e}")
    print("Make sure you're in the correct directory and flextok is properly installed")
    sys.exit(1)


class TrainedFlexTokEncoder:
    """Wrapper for the trained FlexTok encoder."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.load_model(checkpoint_path)
        self.setup_preprocessors()
    
    def load_model(self, checkpoint_path: str):
        """Load the trained model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            # Lightning checkpoint
            state_dict = checkpoint['state_dict']
        else:
            # Direct state dict
            state_dict = checkpoint
        
        # Extract encoder and preprocessor components
        self.encoder = FlexTransformer(
            input_seq_read_key="patched_embeddings",
            output_seq_write_key="encoded_tokens",
            dim=768,  # Default dimension, adjust based on your training
            depth=12,  # Default depth, adjust based on your training
            head_dim=64,
            mlp_ratio=4.0,
            drop=0.1,
            drop_path_rate=0.1,
            use_act_checkpoint=False  # Disable for inference
        )
        
        # Load encoder weights
        encoder_state_dict = {
            k.replace('encoder.', ''): v 
            for k, v in state_dict.items() 
            if k.startswith('encoder.')
        }
        
        if encoder_state_dict:
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print("Encoder weights loaded successfully")
        else:
            print("No encoder weights found in checkpoint")
        
        # Load other components
        self.linear_proj = LinearProjector(
            input_seq_read_key="patches",
            output_seq_write_key="patched_embeddings",
            input_dim=16**2 * 3,  # 16x16 patches, RGB channels
            output_dim=768,
            bias=False
        )
        
        # Load linear projection weights
        proj_state_dict = {
            k.replace('linear_proj.', ''): v 
            for k, v in state_dict.items() 
            if k.startswith('linear_proj.')
        }
        
        if proj_state_dict:
            self.linear_proj.load_state_dict(proj_state_dict, strict=False)
            print("Linear projection weights loaded successfully")
        
        # Move to device
        self.encoder.to(self.device)
        self.linear_proj.to(self.device)
        self.encoder.eval()
        self.linear_proj.eval()
    
    def setup_preprocessors(self):
        """Setup preprocessing pipeline."""
        self.image_patcher = ImagePatcher(
            input_img_read_key="rgb",
            output_patches_write_key="patches",
            patch_size=16,
            flatten_patches=True
        )
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode a single image to tokens."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Prepare batch
        batch = {'rgb': image_tensor}
        
        # Preprocess
        batch = self.image_patcher(batch)
        batch = self.linear_proj(batch)
        
        # Encode
        with torch.no_grad():
            batch = self.encoder(batch)
        
        # Return encoded tokens
        return batch['encoded_tokens']
    
    def encode_batch(self, image_paths: list) -> torch.Tensor:
        """Encode a batch of images to tokens."""
        # Load and preprocess images
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            images.append(image_tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(images).to(self.device)
        batch = {'rgb': batch_tensor}
        
        # Preprocess
        batch = self.image_patcher(batch)
        batch = self.linear_proj(batch)
        
        # Encode
        with torch.no_grad():
            batch = self.encoder(batch)
        
        # Return encoded tokens
        return batch['encoded_tokens']
    
    def get_token_statistics(self, tokens: torch.Tensor) -> dict:
        """Get statistics about the encoded tokens."""
        return {
            'shape': tokens.shape,
            'mean': tokens.mean().item(),
            'std': tokens.std().item(),
            'min': tokens.min().item(),
            'max': tokens.max().item(),
            'num_tokens': tokens.shape[1],
            'token_dim': tokens.shape[2]
        }


def main():
    """Example usage of the trained encoder."""
    # Example checkpoint path (update this to your actual checkpoint)
    checkpoint_path = "./checkpoints/final_model.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or update the checkpoint path")
        return
    
    # Initialize encoder
    encoder = TrainedFlexTokEncoder(checkpoint_path)
    
    # Example: Encode a single image
    example_image_path = "./example_image.jpg"
    
    if os.path.exists(example_image_path):
        print(f"Encoding image: {example_image_path}")
        tokens = encoder.encode_image(example_image_path)
        stats = encoder.get_token_statistics(tokens)
        
        print("Token statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example: Use tokens for downstream tasks
        print(f"\nEncoded tokens shape: {tokens.shape}")
        print(f"Tokens can be used for:")
        print("  - Image classification")
        print("  - Image retrieval")
        print("  - Feature extraction")
        print("  - Downstream vision tasks")
    
    else:
        print(f"Example image not found at {example_image_path}")
        print("Please provide a valid image path or create an example image")
    
    # Example: Encode multiple images
    print("\nExample: Batch encoding")
    example_images = ["./example_image.jpg", "./another_image.jpg"]
    existing_images = [img for img in example_images if os.path.exists(img)]
    
    if len(existing_images) > 1:
        print(f"Batch encoding {len(existing_images)} images")
        batch_tokens = encoder.encode_batch(existing_images)
        batch_stats = encoder.get_token_statistics(batch_tokens)
        
        print("Batch token statistics:")
        for key, value in batch_stats.items():
            print(f"  {key}: {value}")
    
    print("\nEncoder ready for use!")


if __name__ == "__main__":
    main()
