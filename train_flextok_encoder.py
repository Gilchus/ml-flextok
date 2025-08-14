#!/usr/bin/env python3
"""
FlexTok Encoder Finetuning Script with PyTorch Lightning

This script demonstrates how to finetune the FlexTok Encoder component
on custom RGB image datasets using PyTorch Lightning.

Usage:
    python train_flextok_encoder.py --config configs/finetune_encoder.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Add the flextok package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'flextok'))

try:
    from flextok.flextok_wrapper import FlexTok
    from flextok.model.trunks.transformers import FlexTransformer
    from flextok.model.preprocessors.flex_seq_packing import BlockWiseSequencePacker
    from flextok.model.preprocessors.patching import ImagePatcher
    from flextok.model.preprocessors.linear import LinearProjector
    from flextok.model.postprocessors.heads import LinearHead
    from flextok.utils.checkpoint import load_checkpoint
except ImportError as e:
    print(f"Error importing flextok: {e}")
    print("Make sure you're in the correct directory and flextok is properly installed")
    sys.exit(1)


class CustomImageDataset(Dataset):
    """Custom RGB image dataset for FlexTok encoder finetuning."""
    
    def __init__(
        self, 
        data_dir: str, 
        image_size: int = 256,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        max_images: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            self.image_files.extend(list(self.data_dir.rglob(ext)))
        
        if max_images:
            self.image_files = self.image_files[:max_images]
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
            return {
                'rgb': image_tensor,
                'file_path': str(img_path)
            }
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return {
                'rgb': torch.zeros(3, self.image_size, self.image_size),
                'file_path': str(img_path)
            }


class FlexTokEncoderModule(pl.LightningModule):
    """PyTorch Lightning module for FlexTok Encoder finetuning."""
    
    def __init__(
        self,
        # Model architecture
        encoder_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        encoder_head_dim: int = 64,
        encoder_mlp_ratio: float = 4.0,
        
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        
        # Loss parameters
        reconstruction_weight: float = 1.0,
        token_consistency_weight: float = 0.1,
        
        # Data parameters
        image_size: int = 256,
        patch_size: int = 16,
        max_seq_len: int = 1024,
        
        # Regularization
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        
        # Checkpoint loading
        pretrained_encoder_path: Optional[str] = None,
        freeze_vae: bool = True,
        freeze_regularizer: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize components
        self._setup_encoder()
        self._setup_preprocessors()
        self._setup_postprocessors()
        
        # Load pretrained weights if specified
        if pretrained_encoder_path:
            self._load_pretrained_weights(pretrained_encoder_path)
        
        # Freeze components if specified
        if freeze_vae:
            self._freeze_vae()
        if freeze_regularizer:
            self._freeze_regularizer()
    
    def _setup_encoder(self):
        """Setup the FlexTransformer encoder."""
        self.encoder = FlexTransformer(
            input_seq_read_key="patched_embeddings",
            output_seq_write_key="encoded_tokens",
            dim=self.hparams.encoder_dim,
            depth=self.hparams.encoder_depth,
            head_dim=self.hparams.encoder_head_dim,
            mlp_ratio=self.hparams.encoder_mlp_ratio,
            drop=self.hparams.dropout,
            drop_path_rate=self.hparams.drop_path_rate,
            use_act_checkpoint=True,  # Save memory
        )
    
    def _setup_preprocessors(self):
        """Setup preprocessing pipeline."""
        # Image patching
        self.image_patcher = ImagePatcher(
            input_img_read_key="rgb",
            output_patches_write_key="patches",
            patch_size=self.hparams.patch_size,
            flatten_patches=True
        )
        
        # Linear projection to encoder dimension
        patch_dim = self.hparams.patch_size ** 2 * 3  # RGB channels
        self.linear_proj = LinearProjector(
            input_seq_read_key="patches",
            output_seq_write_key="patched_embeddings",
            input_dim=patch_dim,
            output_dim=self.hparams.encoder_dim,
            bias=False
        )
        
        # Sequence packing for efficient training
        self.seq_packer = BlockWiseSequencePacker(
            input_seq_read_key="encoded_tokens",
            output_packed_seq_write_key="packed_tokens",
            max_seq_len=self.hparams.max_seq_len,
            causal=False  # Non-causal for encoder
        )
    
    def _setup_postprocessors(self):
        """Setup postprocessing pipeline."""
        # Project back to patch space for reconstruction loss
        self.reconstruction_head = LinearHead(
            input_seq_read_key="encoded_tokens",
            output_seq_write_key="reconstructed_patches",
            input_dim=self.hparams.encoder_dim,
            output_dim=self.hparams.patch_size ** 2 * 3,
            bias=False
        )
    
    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained encoder weights."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                # Lightning checkpoint
                state_dict = checkpoint['state_dict']
            else:
                # Direct state dict
                state_dict = checkpoint
            
            # Filter encoder weights
            encoder_state_dict = {
                k: v for k, v in state_dict.items() 
                if k.startswith('encoder.')
            }
            
            if encoder_state_dict:
                # Remove 'encoder.' prefix
                clean_state_dict = {
                    k.replace('encoder.', ''): v 
                    for k, v in encoder_state_dict.items()
                }
                missing_keys, unexpected_keys = self.encoder.load_state_dict(
                    clean_state_dict, strict=False
                )
                print(f"Loaded encoder weights from {checkpoint_path}")
                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
            else:
                print(f"No encoder weights found in {checkpoint_path}")
                
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    def _freeze_vae(self):
        """Freeze VAE parameters if available."""
        # This would be implemented if using the full FlexTok wrapper
        pass
    
    def _freeze_regularizer(self):
        """Freeze regularizer parameters if available."""
        # This would be implemented if using the full FlexTok wrapper
        pass
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the encoder pipeline."""
        # Preprocess
        batch = self.image_patcher(batch)
        batch = self.linear_proj(batch)
        
        # Encode
        batch = self.encoder(batch)
        
        # Postprocess
        batch = self.reconstruction_head(batch)
        batch = self.seq_packer(batch)
        
        return batch
    
    def _compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        # Reconstruction loss (L2 between input and reconstructed patches)
        input_patches = batch['patches']
        reconstructed_patches = batch['reconstructed_patches']
        
        # Ensure same shape for loss computation
        if input_patches.shape != reconstructed_patches.shape:
            # Pad or truncate to match
            min_len = min(input_patches.shape[1], reconstructed_patches.shape[1])
            input_patches = input_patches[:, :min_len, :]
            reconstructed_patches = reconstructed_patches[:, :min_len, :]
        
        reconstruction_loss = F.mse_loss(reconstructed_patches, input_patches)
        
        # Token consistency loss (encourage similar patches to have similar encodings)
        # This is a simplified version - you might want to implement more sophisticated
        # consistency measures based on your specific use case
        token_consistency_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = (
            self.hparams.reconstruction_weight * reconstruction_loss +
            self.hparams.token_consistency_weight * token_consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'token_consistency_loss': token_consistency_loss
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Forward pass
        output = self.forward(batch)
        
        # Compute losses
        losses = self._compute_losses(output)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Forward pass
        output = self.forward(batch)
        
        # Compute losses
        losses = self._compute_losses(output)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, prog_bar=True)
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Separate parameter groups for different components
        encoder_params = list(self.encoder.parameters())
        other_params = (
            list(self.linear_proj.parameters()) +
            list(self.reconstruction_head.parameters())
        )
        
        # Different learning rates for different components
        param_groups = [
            {'params': encoder_params, 'lr': self.hparams.learning_rate},
            {'params': other_params, 'lr': self.hparams.learning_rate * 10}  # Higher LR for new components
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in param_groups],
            total_steps=self.hparams.max_steps,
            pct_start=self.hparams.warmup_steps / self.hparams.max_steps,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


class FlexTokDataModule(pl.LightningDataModule):
    """Data module for FlexTok training."""
    
    def __init__(
        self,
        train_data_dir: str,
        val_data_dir: str,
        image_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        max_images: Optional[int] = None
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_images = max_images
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomImageDataset(
                self.train_data_dir,
                image_size=self.image_size,
                max_images=self.max_images
            )
            self.val_dataset = CustomImageDataset(
                self.val_data_dir,
                image_size=self.image_size,
                max_images=self.max_images
            )
    
    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Finetune FlexTok Encoder')
    
    # Data arguments
    parser.add_argument('--train_data_dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--val_data_dir', type=str, required=True,
                       help='Directory containing validation images')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to use (for debugging)')
    
    # Model arguments
    parser.add_argument('--encoder_dim', type=int, default=768,
                       help='Encoder dimension')
    parser.add_argument('--encoder_depth', type=int, default=12,
                       help='Encoder depth')
    parser.add_argument('--encoder_heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for image patching')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--max_steps', type=int, default=100000,
                       help='Maximum number of training steps')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps')
    
    # Checkpoint arguments
    parser.add_argument('--pretrained_encoder_path', type=str, default=None,
                       help='Path to pretrained encoder checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    
    # Logging arguments
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                       help='Log every N steps')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    # Hardware arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--accelerator', type=str, default='auto',
                       help='Accelerator type (auto, gpu, cpu)')
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       help='Training precision')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data module
    data_module = FlexTokDataModule(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images
    )
    
    # Initialize model
    model = FlexTokEncoderModule(
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        encoder_head_dim=args.encoder_dim // args.encoder_heads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        image_size=args.image_size,
        patch_size=args.patch_size,
        pretrained_encoder_path=args.pretrained_encoder_path
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='flextok_encoder-{epoch:02d}-{val_total_loss:.4f}',
            monitor='val_total_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val_total_loss',
            patience=10,
            mode='min'
        )
    ]
    
    # Setup loggers
    loggers = [TensorBoardLogger('logs', name='flextok_encoder')]
    if args.use_wandb:
        loggers.append(WandbLogger(project='flextok-encoder-finetuning'))
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        strategy=DDPStrategy() if args.devices > 1 else 'auto',
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        sync_batchnorm=True if args.devices > 1 else False
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Training completed. Final model saved to {final_checkpoint_path}")


if __name__ == '__main__':
    main()
