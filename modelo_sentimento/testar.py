import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ==========================================
# CONFIGURAÇÕES
# ==========================================
# IMPORTANTE: Aponta para a pasta onde o treino salvou o modelo
MODEL_PATH = "./modelo_final_tp2" 
MAX_LENGTH = 128

def carregar_modelo():
    print("--- Carregando Modelo Treinado ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: A pasta '{MODEL_PATH}' não existe.")
        print("Você precisa rodar o arquivo 'treinar.py' primeiro!")
        exit()
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        exit()

def predict_sentiment(text, model, tokenizer):
    # Prepara o texto
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding=True
    )
    
    # Move para GPU se disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Faz a previsão
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calcula probabilidades
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][prediction].item()
    
    label_map = {0: "NEGATIVO 😡", 1: "POSITIVO 😊"}
    
    print("-" * 50)
    print(f"Review:   {text}")
    print(f"Análise:  {label_map[prediction]}")
    print(f"Certeza:  {confidence:.2%}")
    print("-" * 50)

if __name__ == "__main__":
    # 1. Carrega o modelo (só uma vez no início)
    model, tokenizer = carregar_modelo()
    
    print("\n✅ Modelo carregado com sucesso!")
    print("Digite 'sair' para encerrar.\n")
    
    # 2. Loop de interação
    while True:
        user_input = input("\nDigite uma review de jogo: ")
        
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando...")
            break
            
        if not user_input.strip():
            continue
            
        predict_sentiment(user_input, model, tokenizer)