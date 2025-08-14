#!/usr/bin/env python3
"""
Example script for training FlexTok Encoder with torch tensor datasets

This script demonstrates how to prepare torch tensor data and train the encoder.
"""

import torch
import numpy as np
import os
from pathlib import Path

def create_sample_tensor_dataset(num_samples=1000, image_size=256, channels=3):
    """Create a sample tensor dataset for demonstration."""
    print(f"Creating sample dataset with {num_samples} images of size {image_size}x{image_size}")
    
    # Create random image tensors [N, C, H, W]
    images = torch.randn(num_samples, channels, image_size, image_size)
    
    # Normalize to [0, 1] range (optional)
    images = (images - images.min()) / (images.max() - images.min())
    
    # Create random labels (optional)
    labels = torch.randint(0, 10, (num_samples,))
    
    return images, labels

def save_tensor_dataset(tensors, labels, output_dir, prefix):
    """Save tensor dataset to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tensors
    tensor_path = os.path.join(output_dir, f"{prefix}_tensors.pt")
    torch.save(tensors, tensor_path)
    print(f"Saved tensors to {tensor_path}")
    
    # Save labels
    label_path = os.path.join(output_dir, f"{prefix}_labels.pt")
    torch.save(labels, label_path)
    print(f"Saved labels to {label_path}")
    
    return tensor_path, label_path

def main():
    """Main function to create sample data and demonstrate training."""
    
    # Configuration
    output_dir = "./sample_data"
    num_train = 800
    num_val = 200
    image_size = 256
    channels = 3
    
    print("=== FlexTok Encoder Training with Torch Tensors ===")
    print(f"Creating sample dataset with {num_train} training and {num_val} validation images")
    
    # Create training data
    print("\n1. Creating training data...")
    train_images, train_labels = create_sample_tensor_dataset(
        num_samples=num_train, 
        image_size=image_size, 
        channels=channels
    )
    
    # Create validation data
    print("\n2. Creating validation data...")
    val_images, val_labels = create_sample_tensor_dataset(
        num_samples=num_val, 
        image_size=image_size, 
        channels=channels
    )
    
    # Save datasets
    print("\n3. Saving datasets...")
    train_tensor_path, train_label_path = save_tensor_dataset(
        train_images, train_labels, output_dir, "train"
    )
    val_tensor_path, val_label_path = save_tensor_dataset(
        val_images, val_labels, output_dir, "val"
    )
    
    # Print dataset information
    print(f"\n4. Dataset Information:")
    print(f"Training images: {train_images.shape}")
    print(f"Training labels: {train_labels.shape}")
    print(f"Validation images: {val_images.shape}")
    print(f"Validation labels: {val_labels.shape}")
    
    # Show training commands
    print(f"\n5. Training Commands:")
    print(f"\nBasic training:")
    print(f"python train_flextok_encoder.py \\")
    print(f"    --train_data {train_tensor_path} \\")
    print(f"    --val_data {val_tensor_path} \\")
    print(f"    --train_labels {train_label_path} \\")
    print(f"    --val_labels {val_label_path} \\")
    print(f"    --batch_size 8 \\")
    print(f"    --max_epochs 10")
    
    print(f"\nTraining with custom parameters:")
    print(f"python train_flextok_encoder.py \\")
    print(f"    --train_data {train_tensor_path} \\")
    print(f"    --val_data {val_tensor_path} \\")
    print(f"    --train_labels {train_label_path} \\")
    print(f"    --val_labels {val_label_path} \\")
    print(f"    --encoder_dim 1024 \\")
    print(f"    --encoder_depth 16 \\")
    print(f"    --learning_rate 5e-5 \\")
    print(f"    --batch_size 16 \\")
    print(f"    --max_epochs 50 \\")
    print(f"    --precision 16-mixed")
    
    print(f"\nDebug training (small subset):")
    print(f"python train_flextok_encoder.py \\")
    print(f"    --train_data {train_tensor_path} \\")
    print(f"    --val_data {val_tensor_path} \\")
    print(f"    --max_images 100 \\")
    print(f"    --batch_size 4 \\")
    print(f"    --max_epochs 5")
    
    print(f"\n6. Alternative: Direct tensor input (for testing):")
    print(f"python train_flextok_encoder.py \\")
    print(f"    --train_data 'tensor:({num_train},{channels},{image_size},{image_size})' \\")
    print(f"    --val_data 'tensor:({num_val},{channels},{image_size},{image_size})' \\")
    print(f"    --batch_size 8 \\")
    print(f"    --max_epochs 10")
    
    print(f"\n=== Setup Complete ===")
    print(f"Sample data saved to: {output_dir}")
    print(f"You can now run the training commands above.")
    print(f"Make sure to install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
