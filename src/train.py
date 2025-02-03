import torch
from transformers import (
    MarianMTModel,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import numpy as np
from preprocess import preprocess_dataset
import os
import evaluate

def create_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # BLEU score
        bleu = corpus_bleu(decoded_preds, [decoded_labels])

        # ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(pred, label)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        
        # Average ROUGE scores
        for key in rouge_scores:
            rouge_scores[key] /= len(decoded_preds)

        return {
            'bleu': bleu.score,
            'rouge1_f1': rouge_scores['rouge1'],
            'rouge2_f1': rouge_scores['rouge2'],
            'rougeL_f1': rouge_scores['rougeL']
        }
    return compute_metrics

def train_model(
    tokenizer_name="aubmindlab/bert-base-arabert",
    model_name="Helsinki-NLP/opus-mt-ar-en",
    output_dir="models",
    checkpoint_dir="models/final",  # Using the last saved model
    num_train_epochs=2,  # Training for 2 more epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=250,  # Reduced warmup steps since we're continuing training
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    learning_rate=5e-6  # Further reduced learning rate for final fine-tuning
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset using AraBERT tokenizer
    train_dataset, test_dataset, tokenizer = preprocess_dataset(tokenizer_name)
    if train_dataset is None:
        return

    # Initialize translation model from checkpoint if it exists
    if os.path.exists(checkpoint_dir):
        print(f"Resuming training from checkpoint: {checkpoint_dir}")
        model = MarianMTModel.from_pretrained(checkpoint_dir)
    else:
        print("Starting training from scratch")
        model = MarianMTModel.from_pretrained(model_name)
    
    # Force CPU usage
    model = model.to('cpu')

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=load_best_model_at_end,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        no_cuda=True,  # Force CPU usage
        learning_rate=learning_rate,
        resume_from_checkpoint=checkpoint_dir if os.path.exists(checkpoint_dir) else None
    )

    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=create_compute_metrics(tokenizer)
    )

    # Train the model
    print("Continuing training...")
    trainer.train(resume_from_checkpoint=checkpoint_dir if os.path.exists(checkpoint_dir) else None)

    # Save the final model
    trainer.save_model(f"{output_dir}/final")
    print(f"Model saved to {output_dir}/final")

    # Evaluate the model
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    train_model() 