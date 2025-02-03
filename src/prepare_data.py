import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json
import os

def download_quran_dataset():
    """
    Download the Quran dataset from Tanzil project
    """
    # You can modify this URL based on the specific version you want to use
    url = "https://tanzil.net/trans/en.sahih"
    
    try:
        df = pd.read_csv("data/quran.csv")
        print("Dataset already exists locally")
        return df
    except:
        print("Downloading Quran dataset...")
        # Since direct download might require specific handling, we'll save a placeholder message
        print("Please download the Quran dataset manually from tanzil.net and place it in the data directory")
        print("The file should be in CSV format with columns: surah, ayah, text, translation")
        return None

def preprocess_text(text):
    """
    Preprocess the text by cleaning and normalizing
    """
    if not isinstance(text, str):
        return ""
    
    # Basic preprocessing
    text = text.strip()
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

def prepare_dataset():
    """
    Prepare and split the dataset
    """
    df = download_quran_dataset()
    if df is None:
        return
    
    # Preprocess the text
    df['text'] = df['text'].apply(preprocess_text)
    df['translation'] = df['translation'].apply(preprocess_text)
    
    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the processed datasets
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    prepare_dataset() 