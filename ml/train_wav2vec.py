import os
import torch
import numpy as np
import librosa
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2ForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path

# 1. Setup Data
real_dir = Path("data/real")
fake_dir = Path("data/fake")

# Collect paths and labels
data = {"audio": [], "label": []}

for f in real_dir.glob("*.wav"):
    data["audio"].append(str(f))
    data["label"].append(0) # 0 = Human

for f in fake_dir.glob("*.wav"):
    data["audio"].append(str(f))
    data["label"].append(1) # 1 = AI

dataset = Dataset.from_dict(data)

# 2. Split Dataset
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 3. Initialize Model & Feature Extractor
model_name = "facebook/wav2vec2-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    ignore_mismatched_sizes=True
)

# 4. FREEZE BACKBONE
for param in model.wav2vec2.parameters():
    param.requires_grad = False

# 5. Preprocessing
def preprocess_function(examples):
    audio_arrays = []
    for path in examples["audio"]:
        # librosa.load is robust and doesn't depend on torchcodec
        array, _ = librosa.load(path, sr=16000)
        audio_arrays.append(array)

    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=16000 * 2, # 2 seconds
        truncation=True,
        padding="max_length"
    )
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["audio"])
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["audio"])

# 6. Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 7. Training Config
training_args = TrainingArguments(
    output_dir="./models/checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    push_to_hub=False,
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 9. Train
print("Starting training...")
trainer.train()

# 10. Save Model
output_path = Path("models/production/model_v1.0.0")
output_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_path)
feature_extractor.save_pretrained(output_path)

print(f"\n✅ Model trained and saved to {output_path}")

# Optional: INT8 Quantization for Inference Speed
def quantize_model(model_path):
    # This is a simplified placeholder for post-training quantization
    # In practice, you'd use torch.quantization or optimum
    pass 
