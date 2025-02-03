import torch
from transformers import MarianMTModel, AutoTokenizer
import pandas as pd
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm

def evaluate_model(
    model_path="models/final",
    test_file="data/test.csv",
    batch_size=32,
    max_length=128
):
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(test_file)
    
    # Initialize scorers
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Lists to store predictions and references
    all_predictions = []
    all_references = []
    
    # Generate predictions
    print("Generating predictions...")
    for i in tqdm(range(0, len(test_df), batch_size)):
        batch_texts = test_df['text'].iloc[i:i+batch_size].tolist()
        batch_translations = test_df['translation'].iloc[i:i+batch_size].tolist()
        
        # Tokenize inputs
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate translations
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend(decoded_preds)
        all_references.extend(batch_translations)
    
    # Calculate BLEU score
    bleu = corpus_bleu(all_predictions, [all_references])
    
    # Calculate ROUGE scores
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    for pred, ref in zip(all_predictions, all_references):
        scores = rouge_scorer_obj.score(pred, ref)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeL'] += scores['rougeL'].fmeasure
    
    # Average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= len(all_predictions)
    
    # Calculate word-level F1 score
    def get_words(text):
        return set(text.lower().split())
    
    f1_scores = []
    for pred, ref in zip(all_predictions, all_references):
        pred_words = get_words(pred)
        ref_words = get_words(ref)
        
        # Calculate precision and recall
        common_words = len(pred_words.intersection(ref_words))
        if len(pred_words) > 0:
            precision = common_words / len(pred_words)
        else:
            precision = 0
        
        if len(ref_words) > 0:
            recall = common_words / len(ref_words)
        else:
            recall = 0
        
        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        f1_scores.append(f1)
    
    # Average F1 score
    avg_f1 = np.mean(f1_scores)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"ROUGE-1 F1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rougeL']:.4f}")
    print(f"Word-level F1 Score: {avg_f1:.4f}")
    
    # Save some example predictions
    print("\nSaving example predictions...")
    examples_df = pd.DataFrame({
        'Source': test_df['text'][:5],
        'Reference': test_df['translation'][:5],
        'Prediction': all_predictions[:5]
    })
    examples_df.to_csv('evaluation/example_predictions.csv', index=False)
    print("Example predictions saved to evaluation/example_predictions.csv")
    
    return {
        'bleu': bleu.score,
        'rouge1_f1': rouge_scores['rouge1'],
        'rouge2_f1': rouge_scores['rouge2'],
        'rougeL_f1': rouge_scores['rougeL'],
        'word_f1': avg_f1
    }

if __name__ == "__main__":
    evaluate_model() 