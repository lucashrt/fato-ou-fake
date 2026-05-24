import tkinter as tk
from tkinter import messagebox
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Preparação
nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    text = " ".join([w for w in text.split() if w not in pt_stopwords])
    return text

# Treino do modelo
df = pd.read_csv('base_dados_stf_com_links.csv')
df['text_clean'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text_clean'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# Função chamada pelo botão
def analisar_noticia():
    texto = entrada_texto.get("1.0", tk.END).strip()
    if not texto:
        messagebox.showwarning("Aviso", "Digite um texto para análise.")
        return

    texto_limpo = preprocess_text(texto)
    texto_vec = vectorizer.transform([texto_limpo])
    pred = model.predict(texto_vec)[0]
    prob = max(model.predict_proba(texto_vec)[0])

    resultado.set(f"Resultado: {pred}\nConfiança: {prob:.2%}")

# Interface
janela = tk.Tk()
janela.title("Detector de Fake News - STF")
janela.geometry("500x400")

tk.Label(janela, text="Digite a notícia abaixo:").pack(pady=5)

entrada_texto = tk.Text(janela, height=8, width=55)
entrada_texto.pack(pady=5)

tk.Button(janela, text="Analisar", command=analisar_noticia).pack(pady=10)

resultado = tk.StringVar()
tk.Label(janela, textvariable=resultado, font=("Arial", 12, "bold")).pack(pady=10)

janela.mainloop()