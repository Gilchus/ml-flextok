# FlexTok Encoder Finetuning with Torch Tensors

This repository contains a PyTorch Lightning script for finetuning the FlexTok Encoder component on custom RGB image datasets stored as torch tensors.

## Overview

FlexTok is a state-of-the-art image tokenizer that converts images into flexible-length token sequences. This training script focuses on finetuning just the encoder component, which is useful for:

- **Domain adaptation**: Adapting the encoder to specific image domains (e.g., medical images, satellite imagery)
- **Task-specific finetuning**: Optimizing the encoder for specific downstream tasks
- **Efficient training**: Training only the encoder while keeping other components frozen
- **Direct tensor input**: Works directly with torch tensors instead of image files

## Features

- **Modular Architecture**: Trains only the encoder component while keeping VAE and regularizer frozen
- **PyTorch Lightning**: Modern training framework with built-in distributed training, mixed precision, and logging
- **Flexible Data Loading**: Supports torch tensor datasets with automatic preprocessing
- **Advanced Training Features**: Learning rate scheduling, gradient clipping, early stopping, and checkpointing
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Production Ready**: Includes validation, error handling, and model saving
- **Multiple Formats**: Supports .pt, .pth, .npy, .npz files and direct tensor input

## Installation

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify FlexTok installation**:
   ```bash
   python -c "import flextok; print('FlexTok installed successfully')"
   ```

## Data Preparation

### Supported Tensor Formats

The training script supports multiple tensor formats:

- **PyTorch tensors**: `.pt` or `.pth` files
- **NumPy arrays**: `.npy` files
- **Compressed NumPy**: `.npz` files
- **Direct tensor input**: For testing with random data

### Tensor Requirements

- **Shape**: `[N, C, H, W]` where:
  - `N`: Number of images
  - `C`: Number of channels (3 for RGB)
  - `H`: Image height
  - `W`: Image width
- **Data type**: `torch.float32` or `torch.float16`
- **Value range**: Typically [0, 1] or [-1, 1] (will be normalized automatically)

### Example Data Structure

```python
import torch

# Create sample dataset
num_images = 1000
image_size = 256
channels = 3

# Training data: [N, C, H, W]
train_tensors = torch.randn(num_images, channels, image_size, image_size)
val_tensors = torch.randn(200, channels, image_size, image_size)

# Optional labels
train_labels = torch.randint(0, 10, (num_images,))
val_labels = torch.randint(0, 10, (200,))

# Save to files
torch.save(train_tensors, "train_tensors.pt")
torch.save(val_tensors, "val_tensors.pt")
torch.save(train_labels, "train_labels.pt")
torch.save(val_labels, "val_labels.pt")
```

### Quick Setup with Sample Data

Run the example script to create sample tensor datasets:

```bash
python example_tensor_training.py
```

This will create sample data in `./sample_data/` and show you the training commands.

## Training

### Basic Training Command

```bash
python train_flextok_encoder.py \
    --train_data ./data/train_tensors.pt \
    --val_data ./data/val_tensors.pt \
    --batch_size 8 \
    --max_epochs 100
```

### Training with Labels

```bash
python train_flextok_encoder.py \
    --train_data ./data/train_tensors.pt \
    --val_data ./data/val_tensors.pt \
    --train_labels ./data/train_labels.pt \
    --val_labels ./data/val_labels.pt \
    --batch_size 8 \
    --max_epochs 100
```

### Advanced Training with Custom Parameters

```bash
python train_flextok_encoder.py \
    --train_data ./data/train_tensors.pt \
    --val_data ./data/val_tensors.pt \
    --encoder_dim 1024 \
    --encoder_depth 16 \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --max_epochs 200 \
    --precision 16-mixed \
    --devices 2
```

### Training with Pretrained Weights

```bash
python train_flextok_encoder.py \
    --train_data ./data/train_tensors.pt \
    --val_data ./data/val_tensors.pt \
    --pretrained_encoder_path ./pretrained_encoder.ckpt \
    --learning_rate 1e-5 \
    --batch_size 8
```

### Testing with Random Data

```bash
python train_flextok_encoder.py \
    --train_data 'tensor:(800,3,256,256)' \
    --val_data 'tensor:(200,3,256,256)' \
    --batch_size 8 \
    --max_epochs 10
```

## Configuration

### Command Line Arguments

#### Data Arguments

- `--train_data`: Path to training data tensors (.pt, .pth, .npy, .npz) or "tensor:shape" for testing
- `--val_data`: Path to validation data tensors (.pt, .pth, .npy, .npz) or "tensor:shape" for testing
- `--train_labels`: Path to training labels (optional)
- `--val_labels`: Path to validation labels (optional)
- `--image_size`: Image size for training (default: 256)
- `--batch_size`: Batch size per device (default: 8)
- `--max_images`: Maximum images to use (default: all)
- `--normalize`: Whether to normalize images (default: True)
- `--channel_first`: Whether tensors are in [N, C, H, W] format (default: True)

#### Model Arguments

- `--encoder_dim`: Encoder hidden dimension (default: 768)
- `--encoder_depth`: Number of transformer layers (default: 12)
- `--encoder_heads`: Number of attention heads (default: 12)
- `--patch_size`: Image patch size (default: 16)

#### Training Arguments

- `--learning_rate`: Base learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.01)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--max_steps`: Maximum training steps (default: 100000)
- `--warmup_steps`: Learning rate warmup steps (default: 1000)

#### Hardware Arguments

- `--accelerator`: Accelerator type (auto, gpu, cpu)
- `--devices`: Number of devices to use
- `--precision`: Training precision (16-mixed, 32, 64)
- `--num_workers`: Data loader workers

### Configuration File

You can also use a YAML configuration file:

```bash
python train_flextok_encoder.py --config configs/finetune_encoder.yaml
```

## Model Architecture

The training script implements a simplified FlexTok encoder pipeline:

```
Input Tensor → Image Patching → Linear Projection → Transformer Encoder → Reconstruction Head
```

### Components

1. **ImagePatcher**: Converts images to patches (default: 16x16)
2. **LinearProjector**: Projects patches to encoder dimension
3. **FlexTransformer**: Core transformer encoder with flexible attention
4. **ReconstructionHead**: Projects encoded tokens back to patch space for loss computation

### Loss Function

The training uses a combination of:

- **Reconstruction Loss**: L2 loss between input and reconstructed patches
- **Token Consistency Loss**: Encourages similar patches to have similar encodings

## Training Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/flextok_encoder
```

### Weights & Biases

```bash
python train_flextok_encoder.py --use_wandb
```

### Metrics Tracked

- Training/validation total loss
- Training/validation reconstruction loss
- Training/validation token consistency loss
- Learning rate
- Training time per epoch

## Checkpointing

### Automatic Checkpointing

- **Best Model**: Saves top 3 models based on validation loss
- **Last Model**: Always saves the last model
- **Checkpoint Format**: `flextok_encoder-{epoch:02d}-{val_total_loss:.4f}.ckpt`

### Manual Checkpointing

```python
# Save checkpoint
trainer.save_checkpoint("manual_checkpoint.ckpt")

# Load checkpoint
model = FlexTokEncoderModule.load_from_checkpoint("manual_checkpoint.ckpt")
```

## Inference

After training, use the trained encoder for inference:

```python
from example_usage import TrainedFlexTokEncoder

# Load trained encoder
encoder = TrainedFlexTokEncoder("./checkpoints/final_model.ckpt")

# Encode single image tensor
image_tensor = torch.randn(3, 256, 256)  # [C, H, W]
tokens = encoder.encode_image_tensor(image_tensor)
print(f"Encoded tokens shape: {tokens.shape}")

# Encode batch of image tensors
batch_tensors = torch.randn(4, 3, 256, 256)  # [N, C, H, W]
batch_tokens = encoder.encode_batch_tensors(batch_tensors)
print(f"Batch tokens shape: {batch_tokens.shape}")
```

## Performance Optimization

### Memory Optimization

- **Gradient Checkpointing**: Reduces memory usage at the cost of computation
- **Mixed Precision**: Uses 16-bit precision for faster training
- **Activation Checkpointing**: Built into the FlexTransformer

### Training Speed

- **Distributed Training**: Multi-GPU training with DDP strategy
- **Data Loading**: Multiple workers with pinned memory
- **Batch Size**: Adjust based on your GPU memory

### Recommended Settings

#### Small GPU (8GB VRAM)

```bash
--batch_size 4 --precision 16-mixed --num_workers 2
```

#### Medium GPU (16GB VRAM)

```bash
--batch_size 8 --precision 16-mixed --num_workers 4
```

#### Large GPU (24GB+ VRAM)

```bash
--batch_size 16 --precision 16-mixed --num_workers 8
```

## Troubleshooting

### Common Issues

#### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Reduce image size

#### Import Errors

- Ensure FlexTok is properly installed
- Check Python path includes the flextok directory
- Verify all dependencies are installed

#### Training Instability

- Reduce learning rate
- Increase warmup steps
- Adjust loss weights
- Check data quality

#### Tensor Format Issues

- Ensure tensors are in [N, C, H, W] format
- Check tensor dimensions match expected shapes
- Verify data type is float32 or float16

### Debug Mode

For debugging, use a small subset of data:

```bash
python train_flextok_encoder.py \
    --train_data ./data/train_tensors.pt \
    --val_data ./data/val_tensors.pt \
    --max_images 100 \
    --batch_size 2 \
    --max_epochs 5
```

Or use random data for quick testing:

```bash
python train_flextok_encoder.py \
    --train_data 'tensor:(100,3,256,256)' \
    --val_data 'tensor:(20,3,256,256)' \
    --batch_size 2 \
    --max_epochs 5
```

## Advanced Usage

### Custom Loss Functions

Modify the `_compute_losses` method in `FlexTokEncoderModule`:

```python
def _compute_losses(self, batch):
    # Your custom loss computation
    custom_loss = self.compute_custom_loss(batch)

    return {
        'total_loss': custom_loss,
        'custom_loss': custom_loss
    }
```

### Custom Data Augmentation

Extend the `CustomImageDataset` class:

```python
class CustomImageDataset(Dataset):
    def __init__(self, data_tensors, augment=True):
        # ... existing code ...

        if augment:
            # Add custom tensor augmentations
            self.augment_transform = transforms.Compose([
                # Add your custom tensor transformations
            ])
```

### Multi-GPU Training

```bash
python train_flextok_encoder.py \
    --devices 4 \
    --strategy ddp \
    --batch_size 32
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This training script follows the same license as the FlexTok project. See the main repository for license details.

## Citation

If you use this training script in your research, please cite the FlexTok paper:

```bibtex
@article{flextok2024,
  title={FlexTok: Resampling Images into 1D Token Sequences of Flexible Length},
  author={...},
  journal={...},
  year={2024}
}
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include error messages, system information, and minimal reproduction steps
