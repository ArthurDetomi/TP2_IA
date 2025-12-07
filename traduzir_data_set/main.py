import csv
from deep_translator import GoogleTranslator
from tqdm import tqdm

# ========= CONFIGURAÇÕES =========
ARQUIVO_ENTRADA = "dataset.csv"
ARQUIVO_SAIDA = "reviews_comments_traduzido.csv"
NOME_COLUNA_A_TRADUZIR = "review_text"
COLUNA_SCORE = "review_score"
QTD_POSITIVAS = 500
QTD_NEGATIVAS = 500
# =================================

def cut_string(texto="", tamanho=4999):
    limite = min(len(texto), tamanho)
    resultado = []

    for i in range(limite):
        resultado.append(texto[i])

    return "".join(resultado)


tradutor = GoogleTranslator(source="auto", target="pt")

positivas = []
negativas = []

# ---- 1. Ler o CSV até juntar 500 positivas + 500 negativas ----
with open(ARQUIVO_ENTRADA, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames

    for row in reader:
        score = row[COLUNA_SCORE].strip()

        if score == "1" and len(positivas) < QTD_POSITIVAS:
            positivas.append(row)

        elif score == "-1" and len(negativas) < QTD_NEGATIVAS:
            negativas.append(row)

        # Se já tiver as duas listas completas, pode parar
        if len(positivas) == QTD_POSITIVAS and len(negativas) == QTD_NEGATIVAS:
            break

linhas_selecionadas = positivas + negativas

print(f"Selecionadas {len(positivas)} positivas e {len(negativas)} negativas.")


# ---- 2. Traduzir a coluna desejada com CACHE ----
cache = {}

for row in tqdm(linhas_selecionadas):
    texto_original = row[NOME_COLUNA_A_TRADUZIR]

    if texto_original in cache:
        row[NOME_COLUNA_A_TRADUZIR] = cache[texto_original]
        continue
    
    
    texto_tratado = cut_string(texto_original, 4998)
    
    texto_traduzido = tradutor.translate(texto_tratado)

    cache[texto_original] = texto_traduzido
    row[NOME_COLUNA_A_TRADUZIR] = texto_traduzido


# ---- 3. Salvar novo CSV com as traduções ----
with open(ARQUIVO_SAIDA, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(linhas_selecionadas)

print(f"Arquivo '{ARQUIVO_SAIDA}' criado com sucesso!")
