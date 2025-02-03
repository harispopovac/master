import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset
import os
from farasa.segmenter import FarasaSegmenter

# Initialize Farasa segmenter
farasa_segmenter = FarasaSegmenter(interactive=True)

def preprocess_arabic(text):
    """
    Preprocess Arabic text according to AraBERT paper
    """
    text = str(text)
    text = text.strip()
    text = " ".join(text.split())  # Remove extra whitespace
    # Use Farasa segmentation
    text = farasa_segmenter.segment(text)
    return text

def preprocess_english(text):
    """
    Preprocess English text
    """
    text = str(text)
    text = text.strip()
    text = " ".join(text.split())  # Remove extra whitespace
    return text

class QuranDataset(Dataset):
    def __init__(self, texts, translations, tokenizer, max_length=128):
        self.texts = texts
        self.translations = translations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess_arabic(self.texts[idx])
        translation = preprocess_english(self.translations[idx])

        # Tokenize inputs
        source_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize targets
        target_encoding = self.tokenizer(
            translation,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def preprocess_dataset(model_name="aubmindlab/bert-base-arabert", test_size=0.2, max_length=128):
    """
    Preprocess the Quran dataset and prepare it for training
    """
    # Load the dataset
    try:
        df = pd.read_csv('data/quran.csv')
    except FileNotFoundError:
        print("Dataset not found. Please run download_quran.py first.")
        return None

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = QuranDataset(
        train_df['text'].values,
        train_df['translation'].values,
        tokenizer,
        max_length
    )
    
    test_dataset = QuranDataset(
        test_df['text'].values,
        test_df['translation'].values,
        tokenizer,
        max_length
    )
    
    # Save the splits for later use
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_dataset, test_dataset, tokenizer

if __name__ == "__main__":
    preprocess_dataset() 