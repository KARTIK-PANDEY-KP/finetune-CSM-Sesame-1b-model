# CSM-LoRA: Parameter-Efficient Fine-Tuning for Text-to-Speech Models

This repository contains code for fine-tuning CSM (Conversational Speech Model) is a speech generation model from Sesame using LoRA (Low-Rank Adaptation), a parameter-efficient method that allows fine-tuning large models with minimal memory requirements.

## Installation

```bash
# Clone the repository
git clone https://github.com/KARTIK-PANDEY-KP/finetune-CSM-Sesame-1b-model.git
cd csm-lora

# Install dependencies
pip install -r requirements.txt
```

## Hardware Configuration Options

The training script supports various hardware configurations:

### High-End GPUs (A100, H100, etc.)

For machines with powerful GPUs and ample memory:

```bash
python train_lora.py \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --use_wandb True \
  --use_cpu_fallback False
```

### Mid-Range GPUs (RTX 3090, RTX 4090, etc.)

For consumer-grade GPUs with 24-32GB VRAM:

```bash
python train_lora.py \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --use_wandb True \
  --use_cpu_fallback False
```

### Low-End GPUs (GTX 1080, RTX 2060, etc.)

For GPUs with limited VRAM (8-12GB):

```bash
python train_lora.py \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --use_wandb False \
  --use_cpu_fallback False
```

### CPU-Only Training

For testing or when no GPU is available:

```bash
python train_lora.py \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 \
  --use_wandb False \
  --use_cpu_fallback True
```

## Key Configuration Parameters

You can modify these parameters in the `train_config` dictionary within `train_lora.py`:

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `batch_size` | Number of samples per batch | High-end GPU: 4-8, Mid-range: 2-4, Low-end: 1-2, CPU: 1 |
| `gradient_accumulation_steps` | Number of steps to accumulate gradients before update | Inversely proportional to batch size |
| `learning_rate` | Learning rate for optimizer | 1e-4 to 5e-5 (lower for smaller batches) |
| `num_epochs` | Number of training epochs | 3-10 depending on dataset size |
| `lora_r` | LoRA rank dimension | 8-32 (higher = more capacity, more memory) |
| `lora_alpha` | LoRA alpha scaling factor | Usually 2x `lora_r` |
| `lora_dropout` | Dropout rate for LoRA layers | 0.05-0.1 |
| `use_cpu_fallback` | Use CPU when OOM errors occur | `True` for testing, `False` for production |
| `max_grad_norm` | Maximum gradient norm for clipping | 1.0 (maybe lower for stability) |

## Memory Optimization Tips

1. **Reduce batch size**: This is the most effective way to reduce memory usage
2. **Increase gradient accumulation**: Compensates for smaller batch sizes
3. **Reduce `lora_r`**: Lower rank means fewer parameters to train
4. **Use FP16/BF16**: Enable mixed precision for faster training and lower memory (not included in the current code)
5. **Offload to CPU**: Using `use_cpu_fallback=True` offloads some operations to CPU

## Dataset Preparation

Your dataset should be in JSON format with the following structure:

```json
[
  {
    "text_tokens": [1, 2, 3, ...],
    "audio_tokens": [[1, 2, 3, ...], [4, 5, 6, ...], ...]
  },
  ...
]
```

The script handles both flat and nested audio token structures.

## Monitoring and Checkpoints

- Training progress is displayed with a progress bar
- Model checkpoints are saved after each epoch in the `checkpoints/` directory
- If enabled, training metrics are logged to wandb

## Example: Editing Configuration

To modify the training configuration, edit the `train_config` dictionary in `train_lora.py`:

```python
train_config = {
    "batch_size": 2,               # Adjust based on your GPU
    "learning_rate": 1e-4,
    "num_epochs": 5,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,
    "lora_r": 16,                  # Increase for more capacity
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_wandb": True,             # Set to False if not using wandb
    "checkpoint_dir": "checkpoints",
    "use_cpu_fallback": False      # Set to True for CPU training
}
```

## Troubleshooting

1. **CUDA Out of Memory error**: Try reducing batch size or increasing gradient accumulation
2. **Slow training on CPU**: This is expected; use CPU mode only for testing
3. **Loss not decreasing**: Try adjusting learning rate or check dataset quality
4. **Validation errors**: Make sure vocabulary size matches your dataset

For more information, check the documentation or submit an issue on GitHub.
