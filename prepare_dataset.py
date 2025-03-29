import os
import json
import torch
import torchaudio
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset

def load_llama3_tokenizer():
    """Load the Llama tokenizer used by the model"""
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    return tokenizer

def prepare_dataset():
    # Create directories
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/val", exist_ok=True)
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_csm_1b(device)
    text_tokenizer = load_llama3_tokenizer()
    
    # Download LJSpeech dataset
    print("Downloading LJSpeech dataset...")
    dataset = load_dataset("keithito/lj_speech")
    
    # Split into train and validation
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    # Process train split
    print("Processing training data...")
    train_data = []
    for item in tqdm(train_dataset):
        # Load and resample audio
        audio, sr = torchaudio.load(item["audio"]["path"])
        audio = torchaudio.functional.resample(audio.squeeze(0), orig_freq=sr, new_freq=generator.sample_rate)
        audio = audio.to(device)  # Move audio to correct device
        
        # Tokenize text
        text_tokens = text_tokenizer.encode(item["text"])
        
        # Create audio tokens using the model's audio tokenizer
        audio_tokens = generator._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Create dataset entry
        entry = {
            "text_tokens": text_tokens,
            "audio_tokens": audio_tokens.tolist(),
            "text": item["text"],
            "speaker_id": 0  # LJSpeech has a single speaker
        }
        train_data.append(entry)
        
        # Save every 1000 items to avoid memory issues
        if len(train_data) % 1000 == 0:
            with open("dataset/train/data.json", "w") as f:
                json.dump(train_data, f)
    
    # Save final training data
    with open("dataset/train/data.json", "w") as f:
        json.dump(train_data, f)
    
    # Process validation split
    print("Processing validation data...")
    val_data = []
    for item in tqdm(val_dataset):
        # Load and resample audio
        audio, sr = torchaudio.load(item["audio"]["path"])
        audio = torchaudio.functional.resample(audio.squeeze(0), orig_freq=sr, new_freq=generator.sample_rate)
        audio = audio.to(device)  # Move audio to correct device
        
        # Tokenize text
        text_tokens = text_tokenizer.encode(item["text"])
        
        # Create audio tokens
        audio_tokens = generator._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Create dataset entry
        entry = {
            "text_tokens": text_tokens,
            "audio_tokens": audio_tokens.tolist(),
            "text": item["text"],
            "speaker_id": 0  # LJSpeech has a single speaker
        }
        val_data.append(entry)
    
    # Save validation data
    with open("dataset/val/data.json", "w") as f:
        json.dump(val_data, f)
    
    print(f"Dataset preparation complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

if __name__ == "__main__":
    prepare_dataset() 