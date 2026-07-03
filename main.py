import numpy as np
import pandas as pd
import pyarrow as pa
import os
import zipfile
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from scipy.special import softmax

from datasets import Dataset
from transformers import (
BertTokenizerFast,
AutoModelForSequenceClassification,
TrainingArguments,
Trainer,
pipeline,
BertTokenizer,
BertModel
)

from lime.lime_text import LimeTextExplainer
import shap
from bertviz import head_view

print("NumPy:", np.**version**)
print("Pandas:", pd.**version**)
print("PyArrow:", pa.**version**)

# ===================== DATA LOADING =====================

# Update these paths according to your system

DATA_PATH = "./data"

fake_df = pd.read_csv(os.path.join(DATA_PATH, "Fake.csv"))
true_df = pd.read_csv(os.path.join(DATA_PATH, "True.csv"))

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Duplicate articles:", df.duplicated(subset="text").sum())
df = df.drop_duplicates(subset="text")

print("Combined shape:", df.shape)

df.to_csv(os.path.join(DATA_PATH, "cleaned_fake_news.csv"), index=False)

print(df['label'].value_counts())

# ===================== VISUALIZATION =====================

sns.countplot(x="label", data=df)
plt.title("Fake vs Real Distribution")
plt.show()

df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
print(df["text_length"].describe())

# ===================== TRAIN TEST SPLIT =====================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ===================== DATASET CONVERSION =====================

train_df = pd.DataFrame({"text": X_train, "label": y_train})
val_df = pd.DataFrame({"text": X_test, "label": y_test})

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(val_df)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_fn(batch):
return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
eval_dataset = eval_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ===================== MODEL =====================

model = AutoModelForSequenceClassification.from_pretrained(
"bert-base-uncased",
num_labels=2
)

# ===================== TRAINING =====================

training_args = TrainingArguments(
output_dir="./results",
num_train_epochs=3,
per_device_train_batch_size=8,
per_device_eval_batch_size=8,
weight_decay=0.01,
logging_steps=200,
evaluation_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
metric_for_best_model="accuracy",
report_to="none"
)

def compute_metrics(eval_pred):
logits, labels = eval_pred
predictions = logits.argmax(-1)
acc = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average='weighted')
return {"accuracy": acc, "f1": f1}

trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=eval_dataset,
compute_metrics=compute_metrics
)

trainer.train()

predictions = trainer.predict(eval_dataset)
print(predictions.metrics)

# ===================== SAVE MODEL =====================

MODEL_PATH = "./model"

trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

# ===================== INFERENCE =====================

classifier = pipeline(
"text-classification",
model=MODEL_PATH,
tokenizer=MODEL_PATH
)

texts = [
"Breaking: Government announces new policy today.",
"You won $1,000,000! Click here to claim your prize!",
"Scientists discovered a new planet in the solar system."
]

results = classifier(texts)

label_map = {
"LABEL_0": "Fake",
"LABEL_1": "Real"
}

for text, res in zip(texts, results):
print({
"text": text,
"label": label_map.get(res["label"], res["label"]),
"confidence": res["score"]
})

# ===================== LIME =====================

explainer = LimeTextExplainer(class_names=["Fake", "Real"])

def predictor(texts):
outputs = classifier(texts)
probs = []
for o in outputs:
if o['label'] == 'LABEL_0':
probs.append([o['score'], 1-o['score']])
else:
probs.append([1-o['score'], o['score']])
return np.array(probs)

exp = explainer.explain_instance(
"Government secretly controls media reports.",
predictor,
num_features=10
)

print(exp.as_list())

# ===================== SHAP =====================

shap.initjs()
explainer = shap.Explainer(classifier)

def explain_text(sentence):
shap_values = explainer([sentence])
shap.plots.text(shap_values[0])

# ===================== CONFUSION MATRIX =====================

preds = trainer.predict(eval_dataset)
y_pred = preds.predictions.argmax(axis=1)
y_true = preds.label_ids

cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ===================== ROC CURVE =====================

logits = predictions.predictions
probs = softmax(logits, axis=1)

fpr, tpr, _ = roc_curve(y_true, probs[:,1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = %0.4f" % roc_auc)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
