import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from typing import Optional, Dict, Any
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Wandb not available. Training without logging.")

from models import Model, ModelArgs

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Get max lengths
    max_text_len = max(item["text_tokens"].size(0) for item in batch)
    max_audio_len = max(item["audio_tokens"].size(0) for item in batch)
    
    # Get maximum number of codebooks from all items
    max_codebooks = 1
    for item in batch:
        if len(item["audio_tokens"].shape) > 1:
            max_codebooks = max(max_codebooks, item["audio_tokens"].size(1))
    
    # Initialize tensors with proper dimensions
    text_tokens = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    audio_tokens = torch.zeros(len(batch), max_audio_len, max_codebooks, dtype=torch.long)
    text_mask = torch.zeros(len(batch), max_text_len, dtype=torch.bool)
    audio_mask = torch.zeros(len(batch), max_audio_len, max_codebooks, dtype=torch.bool)
    
    # Fill tensors
    for i, item in enumerate(batch):
        text_len = item["text_tokens"].size(0)
        audio_len = item["audio_tokens"].size(0)
        
        # Fill text tokens and mask
        text_tokens[i, :text_len] = item["text_tokens"]
        text_mask[i, :text_len] = item["text_mask"]
        
        # Handle audio tokens - ensure we get the correct shape
        audio_data = item["audio_tokens"]
        if len(audio_data.shape) == 2:
            # If shape is [seq_len, codebooks], copy each codebook
            codebook_dim = audio_data.size(1)
            audio_tokens[i, :audio_len, :codebook_dim] = audio_data
        elif len(audio_data.shape) == 1:
            # If shape is [seq_len], put in first codebook
            audio_tokens[i, :audio_len, 0] = audio_data
        else:
            raise ValueError(f"Unexpected audio_tokens shape: {audio_data.shape}")
            
        # Handle audio mask
        audio_mask_data = item["audio_mask"]
        if len(audio_mask_data.shape) == 2:
            # If shape is [seq_len, codebooks], copy each codebook
            mask_codebook_dim = audio_mask_data.size(1)
            audio_mask[i, :audio_len, :mask_codebook_dim] = audio_mask_data
        elif len(audio_mask_data.shape) == 1:
            # If shape is [seq_len], repeat for all codebooks
            for j in range(max_codebooks):
                audio_mask[i, :audio_len, j] = audio_mask_data
        else:
            raise ValueError(f"Unexpected audio_mask shape: {audio_mask_data.shape}")
    
    return {
        "text_tokens": text_tokens,
        "audio_tokens": audio_tokens,
        "text_mask": text_mask,
        "audio_mask": audio_mask
    }

class TextToSpeechDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = 2048, vocab_size: int = 1024):
        self.max_length = max_length
        self.vocab_size = vocab_size
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Validate token ranges
        self._validate_tokens()
            
    def __len__(self):
        return len(self.data)
    
    def _validate_tokens(self):
        """Validate that audio tokens are within the acceptable range."""
        max_token = 0
        min_token = float('inf')
        has_nested_lists = False
        
        for item in self.data[:10]:  # Check first 10 items for diagnostics
            audio_tokens = item["audio_tokens"]
            if isinstance(audio_tokens, list):
                # Check if we have nested lists
                if audio_tokens and isinstance(audio_tokens[0], list):
                    has_nested_lists = True
                    # For nested lists, find max/min in each sublist
                    for sublist in audio_tokens:
                        if sublist:  # Only process non-empty sublists
                            curr_max = max(sublist)
                            curr_min = min(sublist)
                            max_token = max(max_token, curr_max)
                            min_token = min(min_token, curr_min)
                elif audio_tokens:  # For flat lists, find max/min directly
                    curr_max = max(audio_tokens)
                    curr_min = min(audio_tokens)
                    max_token = max(max_token, curr_max)
                    min_token = min(min_token, curr_min)
        
        # Only print if we found valid tokens
        if max_token > 0 and min_token < float('inf'):
            print(f"Dataset token range: min={min_token}, max={max_token}, vocab_size={self.vocab_size}")
            print(f"Dataset has {'nested' if has_nested_lists else 'flat'} list structure for audio tokens")
            if max_token >= self.vocab_size:
                print(f"WARNING: Dataset contains tokens ({max_token}) exceeding vocab size ({self.vocab_size})")
        else:
            print("Could not determine token range from the dataset sample")
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Convert text tokens to tensors
        text_tokens = torch.tensor(item["text_tokens"], dtype=torch.long)
        
        # Handle audio tokens based on structure
        audio_tokens = item["audio_tokens"]
        
        # Check if we have nested lists
        if isinstance(audio_tokens, list) and audio_tokens and isinstance(audio_tokens[0], list):
            # Convert the nested list structure to tensor [seq_len, num_codebooks]
            # First find max length of all sublists
            max_len = max(len(sublist) for sublist in audio_tokens)
            num_codebooks = len(audio_tokens)
            
            # Create padded tensor for audio tokens
            audio_tensor = torch.zeros(max_len, num_codebooks, dtype=torch.long)
            
            # Fill the tensor with tokens from each codebook
            for i, codebook in enumerate(audio_tokens):
                audio_tensor[:len(codebook), i] = torch.tensor(codebook, dtype=torch.long)
            
            # Transpose to have shape [sequence_length, codebooks]
            audio_tokens_tensor = audio_tensor
        else:
            # Simple flat list, convert directly
            audio_tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long)
        
        # Validate and clamp audio tokens to vocab size
        if audio_tokens_tensor.max() >= self.vocab_size:
            audio_tokens_tensor = torch.clamp(audio_tokens_tensor, 0, self.vocab_size - 1)
        
        # Truncate if necessary
        if len(text_tokens) > self.max_length:
            text_tokens = text_tokens[:self.max_length]
        
        if len(audio_tokens_tensor.shape) == 1:
            # 1D tensor [seq_len]
            if audio_tokens_tensor.size(0) > self.max_length:
                audio_tokens_tensor = audio_tokens_tensor[:self.max_length]
        else:
            # 2D tensor [seq_len, codebooks]
            if audio_tokens_tensor.size(0) > self.max_length:
                audio_tokens_tensor = audio_tokens_tensor[:self.max_length, :]
        
        # Create masks
        text_mask = torch.ones_like(text_tokens, dtype=torch.bool)
        
        if len(audio_tokens_tensor.shape) == 1:
            # 1D tensor [seq_len]
            audio_mask = torch.ones(audio_tokens_tensor.size(0), dtype=torch.bool)
        else:
            # 2D tensor [seq_len, codebooks]
            audio_mask = torch.ones(audio_tokens_tensor.size(0), audio_tokens_tensor.size(1), dtype=torch.bool)
        
        return {
            "text_tokens": text_tokens,
            "audio_tokens": audio_tokens_tensor,
            "text_mask": text_mask,
            "audio_mask": audio_mask
        }

def setup_lora(model: Model, config: Dict[str, Any]) -> Model:
    """Setup LoRA configuration for the model."""
    # Ensure model config is PEFT-compatible
    if not hasattr(model.config, 'model_type'):
        model.config.model_type = "llama"
    if not hasattr(model.config, 'torch_dtype'):
        model.config.torch_dtype = torch.float32
    
    # Don't override use_cache if it's already set
    if not hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
    
    # Check if CPU fallback is enabled for smaller memory footprint
    use_cpu = config.get("use_cpu_fallback", False)
    
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "text_embeddings", "audio_embeddings",
            "projection", "codebook0_head"
        ],
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    
    # Prepare model for LoRA
    if use_cpu:
        # Skip int8 for CPU training as it can cause more issues
        print("Skipping int8 training setup for CPU training")
    else:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, lora_config)
    
    # For CPU training, manually free memory if possible
    if use_cpu:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model

class LoRATrainer:
    def __init__(
        self,
        model: Model,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Dict[str, Any] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        
        # Training parameters
        self.batch_size = config.get("batch_size", 4)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.num_epochs = config.get("num_epochs", 3)
        self.warmup_steps = config.get("warmup_steps", 100)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_cpu_fallback = config.get("use_cpu_fallback", False)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.use_cpu_fallback else "cpu")
        if self.device.type == "cpu" and not self.use_cpu_fallback:
            print("Warning: CUDA not available, falling back to CPU. This will be very slow.")
        elif self.device.type == "cpu" and self.use_cpu_fallback:
            print("Using CPU for training as specified in configuration.")
        
        self.model.to(self.device)
        
        # Setup dataloaders with custom collate function
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0 if self.device.type == "cpu" else 4,  # Reduce workers on CPU
            collate_fn=collate_fn
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0 if self.device.type == "cpu" else 4,  # Reduce workers on CPU
                collate_fn=collate_fn
            )
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=len(self.train_loader) * self.num_epochs
        )
        
        # Initialize wandb if configured and available
        if config.get("use_wandb", False) and WANDB_AVAILABLE:
            wandb.init(project="csm-lora", config=config)
    
    def train(self):
        """Main training loop"""
        self.model.train()
        
        # Get vocabulary size for validation
        vocab_size = self.model.config.audio_vocab_size
        print(f"Model vocabulary size: {vocab_size}")
        print(f"Training on {self.device}")
        
        # No need to set up caches for training mode
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device - add error handling for OOM
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and not self.use_cpu_fallback:
                        print("\nCUDA OOM error. Consider using CPU fallback for testing by setting use_cpu_fallback=True")
                        raise
                    else:
                        raise
                
                # Debug token values only on first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"Audio tokens shape: {batch['audio_tokens'].shape}")
                    print(f"Audio tokens min: {batch['audio_tokens'].min().item()}, max: {batch['audio_tokens'].max().item()}")
                
                # Use the simplified forward pass for training
                if hasattr(self.model, 'training_forward'):
                    # Ensure audio tokens are valid (within vocabulary size)
                    if batch["audio_tokens"].max() >= vocab_size:
                        if batch_idx == 0 and epoch == 0:  # Only print warning once
                            print(f"Warning: Found tokens outside vocabulary range. Max: {batch['audio_tokens'].max().item()}, Vocab size: {vocab_size}")
                    
                    # Get input and target sequences (for next token prediction)
                    # Handle different shapes of audio tokens
                    if len(batch["audio_tokens"].shape) == 3:
                        # [batch, seq_len, codebooks]
                        # Use only first codebook for now
                        inputs = batch["audio_tokens"][:, :-1, 0]  # Remove last token
                        targets = batch["audio_tokens"][:, 1:, 0]  # Remove first token
                        mask = batch["audio_mask"][:, :-1, 0] if batch["audio_mask"].size(1) > inputs.size(1) else batch["audio_mask"][:, :, 0]
                    else:
                        # [batch, seq_len]
                        inputs = batch["audio_tokens"][:, :-1]  # Remove last token
                        targets = batch["audio_tokens"][:, 1:]  # Remove first token
                        mask = batch["audio_mask"][:, :-1] if batch["audio_mask"].size(1) > inputs.size(1) else batch["audio_mask"]
                    
                    # Ensure targets are within vocabulary range
                    if targets.max() >= vocab_size:
                        if batch_idx == 0 and epoch == 0:  # Only print warning once
                            print(f"Before clamping - Target max: {targets.max().item()}, Target min: {targets.min().item()}")
                        targets = torch.clamp(targets, 0, vocab_size-1)
                    
                    # Forward pass with OOM handling
                    try:
                        outputs = self.model.training_forward(
                            input_ids=inputs,
                            attention_mask=mask,
                            input_pos=torch.arange(inputs.size(1), device=self.device).unsqueeze(0)
                        )
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e) and not self.use_cpu_fallback:
                            print("\nCUDA OOM during forward pass. Try reducing batch size further or use CPU.")
                            raise
                        else:
                            raise
                else:
                    # Fallback to original method
                    outputs = self.model(
                        input_ids=batch["audio_tokens"],
                        attention_mask=batch["audio_mask"],
                        input_pos=torch.arange(batch["audio_tokens"].size(1), device=self.device).unsqueeze(0)
                    )
                    targets = batch["audio_tokens"]
                
                # Calculate loss
                loss = self.calculate_training_loss(outputs, targets)
                
                # Backward pass with gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Handle OOM during backward pass
                try:
                    loss.backward()
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and not self.use_cpu_fallback:
                        print("\nCUDA OOM during backward pass. Try increasing gradient accumulation.")
                        raise
                    else:
                        raise
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                if self.config.get("use_wandb", False) and WANDB_AVAILABLE:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
            
            # Validation if dataset is provided
            if self.val_dataset:
                val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f}")
                if self.config.get("use_wandb", False) and WANDB_AVAILABLE:
                    wandb.log({"val_loss": val_loss})
            
            # Save checkpoint
            self.save_checkpoint(epoch)
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_val_loss = 0
        
        # Get vocabulary size for validation
        vocab_size = self.model.config.audio_vocab_size
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device - add error handling for OOM
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and not self.use_cpu_fallback:
                        print("\nCUDA OOM error in validation. Consider using CPU fallback.")
                        # Return a dummy loss value instead of crashing
                        return float('inf')
                    else:
                        raise
                
                # Use the simplified forward pass for validation too
                if hasattr(self.model, 'training_forward'):
                    # Get input and target sequences (for next token prediction)
                    # Handle different shapes of audio tokens
                    if len(batch["audio_tokens"].shape) == 3:
                        # [batch, seq_len, codebooks]
                        # Use only first codebook for now
                        inputs = batch["audio_tokens"][:, :-1, 0]  # Remove last token
                        targets = batch["audio_tokens"][:, 1:, 0]  # Remove first token
                        mask = batch["audio_mask"][:, :-1, 0] if batch["audio_mask"].size(1) > inputs.size(1) else batch["audio_mask"][:, :, 0]
                    else:
                        # [batch, seq_len]
                        inputs = batch["audio_tokens"][:, :-1]  # Remove last token
                        targets = batch["audio_tokens"][:, 1:]  # Remove first token
                        mask = batch["audio_mask"][:, :-1] if batch["audio_mask"].size(1) > inputs.size(1) else batch["audio_mask"]
                    
                    # Ensure targets are within vocabulary range
                    if targets.max() >= vocab_size:
                        targets = torch.clamp(targets, 0, vocab_size-1)
                    
                    # Forward pass with OOM handling
                    try:
                        outputs = self.model.training_forward(
                            input_ids=inputs,
                            attention_mask=mask,
                            input_pos=torch.arange(inputs.size(1), device=self.device).unsqueeze(0)
                        )
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e) and not self.use_cpu_fallback:
                            print("\nCUDA OOM during validation forward pass.")
                            # Return a dummy loss value instead of crashing
                            return float('inf')
                        else:
                            raise
                else:
                    # Fallback to original method
                    outputs = self.model(
                        input_ids=batch["audio_tokens"],
                        attention_mask=batch["audio_mask"],
                        input_pos=torch.arange(batch["audio_tokens"].size(1), device=self.device).unsqueeze(0)
                    )
                    targets = batch["audio_tokens"]
                
                loss = self.calculate_training_loss(outputs, targets)
                total_val_loss += loss.item()
        
        self.model.train()
        return total_val_loss / len(self.val_loader)
    
    def calculate_training_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss using next token prediction"""
        # Reshape outputs to [batch_size * seq_len, vocab_size]
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        
        # Get the number of classes (vocabulary size)
        num_classes = outputs_flat.size(-1)
        
        # Flatten targets to 1D for loss calculation
        targets_flat = targets.reshape(-1)
        
        # Validate target values - make sure they're within the valid range
        # Clamp values to be within 0 and num_classes-1
        valid_targets = torch.clamp(targets_flat, 0, num_classes-1)
        
        # Print some validation info
        if not torch.equal(targets_flat, valid_targets):
            invalid_count = (targets_flat != valid_targets).sum().item()
            max_val = targets_flat.max().item()
            min_val = targets_flat.min().item()
            print(f"Warning: Found {invalid_count} invalid target indices. Range: [{min_val}, {max_val}], Valid range: [0, {num_classes-1}]")
        
        # Calculate cross entropy loss with valid targets
        return torch.nn.functional.cross_entropy(outputs_flat, valid_targets)
    
    def calculate_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the loss for the model outputs"""
        # Handle different dimensionalities of outputs and targets
        target = batch["audio_tokens"]
        
        # Reshape outputs to [batch_size * seq_len, vocab_size]
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        
        # Flatten audio tokens to 1D for loss calculation
        # If tokens are 3D [batch, seq, codebooks], we need to handle it
        if len(target.shape) == 3:
            # For multiple codebooks, we can either:
            # 1. Take one codebook (e.g., the first one)
            target_flat = target[:, :, 0].reshape(-1)
            # Or 2. Average predictions across codebooks (more complex)
        else:
            # For single codebook [batch, seq]
            target_flat = target.reshape(-1)
        
        # Calculate cross entropy loss
        return torch.nn.functional.cross_entropy(outputs_flat, target_flat)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"lora_checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        
        # Save LoRA weights separately
        lora_path = os.path.join(checkpoint_dir, f"lora_weights_epoch_{epoch}")
        self.model.save_pretrained(lora_path)

def main():
    # Model configuration
    model_config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=1024,
        audio_num_codebooks=8
    )
    
    # Add PEFT-compatible configuration
    model_config.model_type = "llama"  # Add model_type for PEFT compatibility
    model_config.use_cache = False  # Disable cache for model during training
    model_config.torch_dtype = torch.float32  # Specify dtype
    
    # Training configuration
    train_config = {
        "batch_size": 1,  # Minimum batch size for testing
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "warmup_steps": 100,
        "gradient_accumulation_steps": 16,  # Increased accumulation to compensate
        "max_grad_norm": 1.0,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "use_wandb": False,
        "checkpoint_dir": "checkpoints",
        "use_cpu_fallback": True  # Add CPU fallback option for testing
    }
    
    # Initialize model
    model = Model(model_config)
    
    # Add a simple forward for training
    def training_forward(self, input_ids, attention_mask=None, **kwargs):
        """Simple forward pass for training that avoids generate_frame and caching issues"""
        try:
            # Get the embeddings - handle both 1D and 2D inputs
            if len(input_ids.shape) == 2:
                # [batch_size, seq_len]
                audio_embeds = self.audio_embeddings(input_ids)
            elif len(input_ids.shape) == 3:
                # [batch_size, seq_len, codebooks]
                # We'll use only the first codebook for training for simplicity
                audio_embeds = self.audio_embeddings(input_ids[:, :, 0])
            else:
                raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")
            
            # Pass through backbone (not using cached attention)
            input_pos = kwargs.get('input_pos', torch.arange(audio_embeds.size(1), device=input_ids.device).unsqueeze(0))
            backbone_outputs = self.backbone(audio_embeds, input_pos=input_pos)
            
            # Get logits for next token prediction
            logits = self.codebook0_head(backbone_outputs)
            
            return logits
        except Exception as e:
            print(f"Error in training_forward: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Replace the model's forward method with our training-specific one
    model.training_forward = training_forward.__get__(model)
    
    # Setup LoRA
    model = setup_lora(model, train_config)
    
    # Load datasets
    train_dataset = TextToSpeechDataset(
        "dataset/train/data.json", 
        vocab_size=model_config.audio_vocab_size
    )
    val_dataset = TextToSpeechDataset(
        "dataset/val/data.json", 
        vocab_size=model_config.audio_vocab_size
    )
    
    # Initialize trainer
    trainer = LoRATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 