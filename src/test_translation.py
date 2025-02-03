import torch
from transformers import MarianMTModel, AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
import pandas as pd
import argparse
import sys

def load_model_and_tokenizer(model_path="models/final"):
    """Load the trained model and tokenizer."""
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    model.eval()
    return model, tokenizer

def preprocess_arabic_text(text):
    """Preprocess Arabic text using AraBERT preprocessor."""
    arabert_prep = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabert")
    return arabert_prep.preprocess(text)

def translate_verse(model, tokenizer, arabic_text):
    """Translate a single Arabic verse to English."""
    # Preprocess the input text
    preprocessed_text = preprocess_arabic_text(arabic_text)
    
    # Tokenize
    inputs = tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True
        )
    
    # Decode the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def main():
    parser = argparse.ArgumentParser(description='Translate Arabic text to English')
    parser.add_argument('--input', type=str, help='Input text file path')
    args = parser.parse_args()

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Read input text
        if args.input:
            with open(args.input, 'r', encoding='utf-8') as f:
                arabic_text = f.read().strip()
        else:
            arabic_text = sys.stdin.read().strip()
        
        # Translate
        translation = translate_verse(model, tokenizer, arabic_text)
        
        # Print translation (will be captured by PHP)
        print(translation)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 