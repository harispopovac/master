# AraBERT Qur'an Evaluation Model

This project fine-tunes AraBERT on the Qur'an dataset for analysis tasks and evaluates the model using BLEU, ROUGE, and F1 metrics.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download the Qur'an dataset from [Tanzil.net](https://tanzil.net/trans/) and save it as `data/quran.csv`
2. Run the data preparation script:
```bash
python src/prepare_data.py
```

## Training

To train the model, run:
```bash
python src/train.py
```

The model will be saved in the `models` directory, and training metrics will be logged during the process.

## Project Structure

```
├── data/               # Dataset directory
├── models/            # Saved model checkpoints
├── src/               # Source code
│   ├── prepare_data.py  # Data preparation script
│   └── train.py        # Training script
├── evaluation/        # Evaluation results
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Model Details

- Base model: AraBERT (aubmindlab/arabert-base)
- Training parameters:
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 3
  - Weight decay: 0.01

## Evaluation Metrics

The model is evaluated using:
- BLEU score
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- F1 score

Results are computed during training and logged to the console. 