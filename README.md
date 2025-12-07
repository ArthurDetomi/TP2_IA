# 📊 Análise de Sentimentos para Comentários de Jogos

Este projeto treina e executa um modelo de **análise de sentimentos** para identificar se um comentário sobre um jogo é **positivo** ou **negativo**.
Ele inclui:

- Um script de **treinamento** do modelo (`treinar.py`)
- Um **chat bot** simples que recebe um comentário no terminal e retorna a análise (`main.py`)

O programa utiliza modelos da **Hugging Face**, **PyTorch** e pré-processamento com **pandas**, **NumPy** e **scikit-learn**.

---

## 📦 Instalação das Dependências

Recomenda-se o uso de um ambiente virtual.

Instale todas as dependências necessárias:

```bash
pip install torch
pip install transformers
pip install pandas
pip install numpy
pip install scikit-learn
pip install datasets
pip install accelerate
```

(Se estiver usando GPU AMD/NVIDIA, você pode precisar de versões específicas do PyTorch.)

---

## 🧠 Treinamento do Modelo

Antes de rodar o chatbot, é necessário **treinar o modelo**:

```bash
python3 treinar.py
```

Isso irá:

- Carregar o dataset
- Tokenizar os textos
- Treinar um modelo de classificação de sentimentos
- Salvar o modelo treinado para uso no `main.py`

---

## 💬 Rodar o Chat Bot

Após treinar o modelo, execute:

```bash
python3 main.py
```

O chatbot irá pedir para você digitar comentários sobre jogos.
Para cada comentário, ele retorna:

- O texto analisado
- O sentimento (positivo ou negativo)
- A confiança do modelo

---

## 🧾 Exemplo de Saída

```
-------
Review: O jogo é muito divertido!
Análise:  POSITIVO 😊
Certeza:  94.12%
-------
```
