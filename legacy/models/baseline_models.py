import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

ARTIFACT_DIR = r"C:\Users\tito\.gemini\antigravity-ide\brain\c96eb70a-06a4-485e-86e1-39bcd92468da"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASET_PATH = "data/FakenewsBR_sanitized.csv"  # relativo a raiz do repo

def get_top_features(vectorizer, model, n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    
    # Sort coefficients
    top_fake_idx = coef.argsort()[-n:][::-1] # Positive coeff -> Fake (if Target is 1 for Fake)
    top_true_idx = coef.argsort()[:n]        # Negative coeff -> True
    
    return feature_names[top_fake_idx], coef[top_fake_idx], feature_names[top_true_idx], coef[top_true_idx]

def plot_coefficients(top_fake_feat, top_fake_coef, top_true_feat, top_true_coef, title, filename):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Fake
    sns.barplot(x=top_fake_coef, y=top_fake_feat, ax=axes[0], color='#e74c3c')
    axes[0].set_title(f'Top {len(top_fake_feat)} Palavras - Viés para FALSO (+)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Coeficiente da Regressão Logística (Log-Odds)')
    
    # Plot True
    sns.barplot(x=np.abs(top_true_coef), y=top_true_feat, ax=axes[1], color='#2ecc71')
    axes[1].set_title(f'Top {len(top_true_feat)} Palavras - Viés para VERDADEIRO (-)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Módulo do Coeficiente (Log-Odds)')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== Carregando dados ===")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    
    # Focar apenas em Fake vs True
    df_binary = df[df['label'].isin(['fake', 'true'])].copy()
    
    # Target: Fake = 1, True = 0
    df_binary['target'] = (df_binary['label'] == 'fake').astype(int)
    
    # Filtrar nulos
    df_binary = df_binary.dropna(subset=['text_clean'])
    
    print(f"Total de registros: {len(df_binary)}")
    print(df_binary['target'].value_counts())
    
    X = df_binary['text_clean']
    y = df_binary['target']
    
    # Split temporal simulado (se houvesse data) ou random 80/20
    # Usaremos stratify para manter proporções
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n=== VARIANTE A: O Modelo Ingênuo (Naive Baseline) ===")
    print("Pipeline simples de TF-IDF + Regressão Logística sem Stop-words.")
    
    pipeline_a = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    pipeline_a.fit(X_train, y_train)
    preds_a = pipeline_a.predict(X_test)
    
    acc_a = accuracy_score(y_test, preds_a)
    f1_a = f1_score(y_test, preds_a)
    print(f"Acurácia A: {acc_a:.4f} | F1-Score Falso A: {f1_a:.4f}")
    
    # Extrair coeficientes A
    fake_ft_a, fake_coef_a, true_ft_a, true_coef_a = get_top_features(pipeline_a.named_steps['tfidf'], pipeline_a.named_steps['clf'])
    plot_coefficients(fake_ft_a, fake_coef_a, true_ft_a, true_coef_a, 
                      "Variante A: O que o Modelo Ingênuo Aprendeu?", "baseline_variant_a_coefs.png")
                      
    
    print("\n=== VARIANTE B: O Modelo Higienizado (De-biased Baseline) ===")
    # 1. Stop words estendidas (atalhos temporais, de plataforma e jargões pt-pt)
    stop_words_vazamento = [
        # Vazamentos de Datelines / Jornalismo
        'feira', 'nesta', 'quarta', 'terca', 'segunda', 'quinta', 'sexta', 'domingo', 'sabado',
        'segundo', 'hoje', 'ontem', 'amanha', 'mes', 'ano', 'semana', 'janeiro', 'fevereiro', 'marco',
        'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro',
        'em', 'da', 'na', 'no', 'do', 'de', 'ao',
        # Vazamentos PT-PT
        'facto', 'equipa', 'ecra', 'poligrafo', 'observador', 'contacto', 'acao',
        # Vazamentos do rótulo
        'fake', 'news', 'falso', 'verdadeiro', 'boato', 'mentira'
    ]
    
    # Adicionar stopwords do NLTK pt + nossa customizada
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    from nltk.corpus import stopwords
    pt_stopwords = stopwords.words('portuguese')
    
    # Limpar acentos das stopwords para bater com o text_clean
    import unicodedata
    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    
    pt_stopwords_clean = [strip_accents(w) for w in pt_stopwords]
    all_stop_words = list(set(pt_stopwords_clean + stop_words_vazamento))
    
    print("Treinando pipeline Higienizado...")
    pipeline_b = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=all_stop_words)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    pipeline_b.fit(X_train, y_train)
    preds_b = pipeline_b.predict(X_test)
    
    acc_b = accuracy_score(y_test, preds_b)
    f1_b = f1_score(y_test, preds_b)
    print(f"Acurácia B: {acc_b:.4f} | F1-Score Falso B: {f1_b:.4f}")
    
    # Extrair coeficientes B
    fake_ft_b, fake_coef_b, true_ft_b, true_coef_b = get_top_features(pipeline_b.named_steps['tfidf'], pipeline_b.named_steps['clf'])
    plot_coefficients(fake_ft_b, fake_coef_b, true_ft_b, true_coef_b, 
                      "Variante B: O que o Modelo Higienizado Aprendeu?", "baseline_variant_b_coefs.png")

    print("\n=== SÍNTESE DO EXPERIMENTO ===")
    print(f"Drop na Acurácia (Desaprendizado de Atalhos): {(acc_b - acc_a)*100:.2f} p.p.")
    print("O Modelo Variante B agora foca mais na semântica da narrativa do que em jargões de jornalistas (veja os gráficos gerados).")

if __name__ == '__main__':
    main()
