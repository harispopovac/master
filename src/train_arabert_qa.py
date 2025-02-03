import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset, Dataset
import numpy as np
import evaluate
import os
from typing import Dict, List
import json
import re

def clean_arabic_text(text: str) -> str:
    """
    Clean and normalize Arabic text
    """
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # Remove tashkeel
    text = re.sub(r'\u0640', '', text)  # Remove tatweel
    text = re.sub(r'[إأٱآا]', 'ا', text)  # Normalize alef
    text = re.sub(r'[ؤئ]', 'ء', text)  # Normalize hamza
    text = re.sub(r'ة', 'ه', text)  # Normalize teh marbuta
    text = re.sub(r'[يى]', 'ي', text)  # Normalize yeh
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

def preprocess_qa_dataset(
    examples: Dict,
    tokenizer,
    max_length: int = 384,
    stride: int = 128,
    pad_to_max_length: bool = True
):
    """
    Preprocess the dataset for question answering.
    """
    # Clean and normalize Arabic text
    questions = [clean_arabic_text(q.strip()) for q in examples["question"]]
    contexts = [clean_arabic_text(c.strip()) for c in examples["context"]]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        stride=stride,
        padding="max_length" if pad_to_max_length else False,
        truncation="only_second",  # Truncate only the context if needed
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    # Since one example might give us several features if it has a long context
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    # Let's label those examples
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        sequence_ids = tokenized.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Get the cleaned answer text and context
        cleaned_answer = clean_arabic_text(examples["answer_text"][sample_mapping[i]])
        cleaned_context = contexts[sample_mapping[i]]
        
        # Find the answer in the cleaned context
        answer_start = cleaned_context.find(cleaned_answer)

        # If the answer is not fully inside the context, label is (0, 0)
        if answer_start == -1:
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            # Otherwise it's the start and end token positions
            answer_end = answer_start + len(cleaned_answer)

            # Back to token map
            token_start = token_end = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start:
                token_start += 1
            token_start -= 1

            while token_end <= context_end and offsets[token_end][1] <= answer_end:
                token_end += 1
            token_end -= 1

            tokenized["start_positions"].append(token_start)
            tokenized["end_positions"].append(token_end)

    return tokenized

def compute_metrics(eval_pred):
    """
    Compute metrics for question answering using simple start/end position accuracy
    """
    predictions, labels = eval_pred
    start_logits, end_logits = predictions
    
    # Convert to numpy for easier handling
    start_logits = start_logits.argmax(-1)
    end_logits = end_logits.argmax(-1)
    
    # Labels are already in the correct format
    start_positions = labels[0]
    end_positions = labels[1]
    
    # Calculate accuracy for start and end positions
    start_accuracy = (start_logits == start_positions).mean()
    end_accuracy = (end_logits == end_positions).mean()
    
    # Calculate F1 as average of start and end accuracy
    f1 = (start_accuracy + end_accuracy) / 2
    
    return {
        'start_accuracy': float(start_accuracy),
        'end_accuracy': float(end_accuracy),
        'f1': float(f1)
    }

def load_quran_qa_dataset(data_dir: str = "data/qa", max_examples: int = None):
    """
    Load the Quran QA dataset from JSON files
    max_examples: if set, only load this many examples for faster testing
    """
    train_data = []
    test_data = []
    
    # Load training data
    with open(os.path.join(data_dir, "train.json"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= int(max_examples * 0.9):  # 90% for training
                break
            try:
                example = json.loads(line.strip())
                train_data.append(example)
            except json.JSONDecodeError:
                continue
    
    # Load test data
    with open(os.path.join(data_dir, "test.json"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= int(max_examples * 0.1):  # 10% for testing
                break
            try:
                example = json.loads(line.strip())
                test_data.append(example)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")
    
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    return train_dataset, test_dataset

def train_arabert_qa(
    model_name: str = "aubmindlab/bert-base-arabertv2",  # Correct model name
    output_dir: str = "models/arabert_qa",
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    learning_rate: float = 5e-5,
    max_length: int = 256,
    data_dir: str = "data/qa",
    max_examples: int = 1000
):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Force CPU
    device = torch.device("cpu")
    model = model.to(device)

    # Load Quran QA dataset
    train_dataset, test_dataset = load_quran_qa_dataset(data_dir, max_examples)
    
    # Preprocess the datasets
    tokenized_train = train_dataset.map(
        lambda x: preprocess_qa_dataset(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=1
    )
    
    tokenized_test = test_dataset.map(
        lambda x: preprocess_qa_dataset(x, tokenizer, max_length),
        batched=True,
        remove_columns=test_dataset.column_names,
        num_proc=1
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=5,  # More frequent logging
        evaluation_strategy="steps",  # Evaluate more frequently
        eval_steps=50,  # Evaluate every 50 steps
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        gradient_accumulation_steps=2,  # Reduced gradient accumulation
        save_total_limit=2,
        remove_unused_columns=True,
        dataloader_num_workers=0,
        group_by_length=True,
        report_to="none",
        no_cuda=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(f"{output_dir}/final")
    print(f"Model saved to {output_dir}/final")

if __name__ == "__main__":
    train_arabert_qa() 