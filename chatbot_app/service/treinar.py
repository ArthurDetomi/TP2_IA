import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset

# ==========================================
# CONFIGURAÇÕES
# ==========================================
MODEL_NAME = "neuralmind/bert-base-portuguese-cased" 
OUTPUT_DIR = "./results_tp2"
FINAL_MODEL_DIR = "../data/modelo_final_tp2"
NUM_LABELS = 2 
MAX_LENGTH = 128
DATASET_PATH = "./data/dataset.csv"

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
def load_and_preprocess_data():
    print("--- Carregando Dataset ---")
    filename = DATASET_PATH
    
    try:
        df = pd.read_csv(filename, on_bad_lines='skip') 
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{filename}' não encontrado.")
        return None, None

    # Filtros e Limpeza (Mesma lógica anterior)
    if 'language' in df.columns:
        df = df[df['language'].str.contains('portuguese', case=False, na=False)]

    if 'review_text' in df.columns: pass
    else:
        for col in ['review', 'text', 'body']:
            if col in df.columns:
                df = df.rename(columns={col: 'review_text'})
                break
    
    if 'review_score' in df.columns:
        df = df.rename(columns={'review_score': 'labels'})
    elif 'score' in df.columns: 
         df = df.rename(columns={'score': 'labels'})
    
    df = df.dropna(subset=['review_text', 'labels'])
    df['review_text'] = df['review_text'].astype(str)
    df['labels'] = df['labels'].replace({-1: 0})
    df['labels'] = pd.to_numeric(df['labels'], errors='coerce')
    df = df.dropna(subset=['labels'])
    df['labels'] = df['labels'].astype(int)
    df = df[df['labels'].isin([0, 1])]

    # Amostragem (1000 amostras)
    MAX_SAMPLES = 1000
    if len(df) > MAX_SAMPLES:
        print(f"[AMOSTRAGEM] Reduzindo para {MAX_SAMPLES} amostras...")
        pos_df = df[df['labels'] == 1]
        neg_df = df[df['labels'] == 0]
        
        quota = MAX_SAMPLES // 2
        take_pos = min(len(pos_df), quota)
        take_neg = min(len(neg_df), quota)
        
        # Completa com o que sobrar
        remainder = MAX_SAMPLES - (take_pos + take_neg)
        if remainder > 0:
            if len(pos_df) > take_pos: take_pos += min(remainder, len(pos_df) - take_pos)
            elif len(neg_df) > take_neg: take_neg += min(remainder, len(neg_df) - take_neg)

        sampled_pos = pos_df.sample(n=take_pos, random_state=42)
        sampled_neg = neg_df.sample(n=take_neg, random_state=42)
        df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total Final: {len(df)} (Pos: {len(df[df['labels']==1])}, Neg: {len(df[df['labels']==0])})")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["review_text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# ==========================================
# 3. EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    print(f"--- Iniciando Script de Treinamento ---")
    
    train_dataset, test_dataset = load_and_preprocess_data()
    
    if train_dataset is None:
        exit()

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    use_fp16 = torch.cuda.is_available()
    print(f"GPU Disponível: {use_fp16}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=use_fp16,
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print("\n--- Resultados Finais ---")
    print(results)
    
    print(f"\n--- Salvando modelo em '{FINAL_MODEL_DIR}' ---")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("✅ Treinamento concluído! Agora você pode rodar o arquivo 'testar.py'.")