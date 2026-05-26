import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
pt_stopwords = stopwords.words('portuguese')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    text = " ".join([word for word in text.split() if word not in pt_stopwords])
    return text

# 1. Carregar a base de dados do STF
df = pd.read_csv('base_dados_stf.csv')

# 2. Aplicar Pré-processamento
df['text_clean'] = df['text'].apply(preprocess_text)

# 3. Divisão da base (70% Treino, 30% Teste)
X = df['text_clean']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 4. Vetorização (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Treinamento do Modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Avaliação e Validação
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"--- RESULTADOS DO MODELO (TEMA: STF) ---")
print(f"Acurácia Geral: {accuracy:.2%}")
print("\nRelatório de Métricas:")
print(classification_report(y_test, y_pred))

# Gerar Matriz de Confusão
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Fake', 'Fato'], yticklabels=['Fake', 'Fato'])
plt.xlabel('Predição do Modelo')
plt.ylabel('Valor Real (Gabarito)')
plt.title('Matriz de Confusão: Classificação de Notícias STF')
plt.savefig('matriz_stf.png')

# 7. Demonstração com Novos Exemplos
print("\n--- TESTE DE NOVAS ENTRADAS (DEMONSTRAÇÃO) ---")
testes = [
    "STF decide que todos os brasileiros devem pagar nova taxa de internet em 2026.",
    "O Supremo Tribunal Federal manteve a decisão sobre o ICMS em sessão plenária.",
    "Uma nova decisão do STF anula todos os concursos públicos realizados nos últimos 10 anos devido a supostas irregularidades no processo."
]
testes_clean = [preprocess_text(t) for t in testes]
testes_vec = vectorizer.transform(testes_clean)
preds = model.predict(testes_vec)

for t, p in zip(testes, preds):
    print(f"Texto: {t}")
    print(f"Resultado: {p}\n")
