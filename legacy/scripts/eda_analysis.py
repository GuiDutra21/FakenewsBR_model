import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Create plots folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plots_dir = os.path.join(ROOT, "plots")
os.makedirs(plots_dir, exist_ok=True)

# 1. Load Sanitized Data
file_path = os.path.join(ROOT, "data", "FakenewsBR_sanitized.csv")
print("Loading sanitized dataset...")
df = pd.read_csv(file_path, low_memory=False)
print(f"Loaded {len(df)} rows.")

# 2. Text Length & Distribution Analysis
print("\n=== 1. Text Length Analysis ===")
length_stats_class = df.groupby('label')[['char_len', 'word_len']].describe()
print("Length stats by class:")
print(length_stats_class.to_string())

length_stats_subset = df.groupby('dataset_name')[['char_len', 'word_len']].agg(['mean', 'median', 'std'])
print("\nLength stats by subset:")
print(length_stats_subset.to_string())

length_stats_source = df.groupby('source_type')[['char_len', 'word_len']].agg(['mean', 'median', 'std'])
print("\nLength stats by source type:")
print(length_stats_source.to_string())

# Plot: Text length distributions (Log scale for words)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x='label', y='word_len', hue='label', palette={'fake': '#e74c3c', 'true': '#2ecc71'}, ax=axes[0], showfliers=False)
axes[0].set_title("Distribuição do Número de Palavras (sem outliers visuais)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Classe", fontsize=11)
axes[0].set_ylabel("Número de Palavras", fontsize=11)

sns.kdeplot(data=df, x='word_len', hue='label', common_norm=False, log_scale=True, palette={'fake': '#e74c3c', 'true': '#2ecc71'}, ax=axes[1], fill=True, alpha=0.3)
axes[1].set_title("Densidade de Palavras por Texto (Escala Logarítmica)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Número de Palavras (log)", fontsize=11)
axes[1].set_ylabel("Densidade", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "length_distribution_by_class.png"), dpi=200)
plt.close()

# Plot: Word length by subset
fig, ax = plt.subplots(figsize=(12, 6))
order = df.groupby('dataset_name')['word_len'].median().sort_values().index
sns.boxplot(data=df, x='dataset_name', y='word_len', hue='label', order=order, palette={'fake': '#e74c3c', 'true': '#2ecc71'}, showfliers=False, ax=ax)
ax.set_title("Tamanho dos Textos por Subconjunto e Classe (Mediana ordenada)", fontsize=13, fontweight='bold')
ax.set_xlabel("Subconjunto", fontsize=11)
ax.set_ylabel("Número de Palavras", fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "length_distribution_by_subset.png"), dpi=200)
plt.close()


# 3. Vocabulary Richness & Type-Token Ratio (TTR)
print("\n=== 2. Vocabulary Richness & TTR ===")
def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'\b[a-záàâãéèêíïóôõöúçñ]+\b', text.lower())

def vocab_metrics(texts):
    tokens = []
    for t in texts:
        tokens.extend(tokenize(t))
    total_tokens = len(tokens)
    counts = Counter(tokens)
    unique_types = len(counts)
    ttr = (unique_types / total_tokens * 100) if total_tokens > 0 else 0
    hapax = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = (hapax / unique_types * 100) if unique_types > 0 else 0
    return {
        'total_tokens': total_tokens,
        'unique_types': unique_types,
        'ttr_%': round(ttr, 2),
        'hapax_count': hapax,
        'hapax_%': round(hapax_ratio, 2)
    }

metrics_class = {}
for label in ['fake', 'true']:
    metrics_class[label] = vocab_metrics(df[df['label'] == label]['text_clean'])
print("Vocab metrics by class:")
print(pd.DataFrame(metrics_class))

metrics_subset = {}
for subset in df['dataset_name'].unique():
    metrics_subset[subset] = vocab_metrics(df[df['dataset_name'] == subset]['text_clean'])
print("\nVocab metrics by subset:")
print(pd.DataFrame(metrics_subset).T)


# 4. Portuguese Stopwords & N-Gram Analysis
print("\n=== 3. N-gram Frequency Analysis ===")

PT_STOPWORDS = {
    'a', 'à', 'adeus', 'agora', 'aí', 'ainda', 'além', 'algo', 'alguém', 'algum', 'alguma', 'algumas', 'alguns',
    'ali', 'ampla', 'amplas', 'amplo', 'amplos', 'ano', 'anos', 'ante', 'antes', 'ao', 'aos', 'apenas', 'apoio',
    'após', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'área', 'as', 'às', 'assim', 'até', 'atrás',
    'através', 'baixo', 'bastante', 'bem', 'boa', 'boas', 'bom', 'bons', 'breve', 'cá', 'cada', 'cento', 'cerca',
    'certeza', 'cima', 'cinco', 'coisa', 'coisas', 'com', 'como', 'conselho', 'contra', 'contudo', 'custa', 'da',
    'dá', 'dão', 'daquela', 'daquelas', 'daquele', 'daqueles', 'dar', 'das', 'de', 'debaixo', 'dela', 'delas',
    'dele', 'deles', 'demais', 'depois', 'desde', 'dessa', 'dessas', 'desse', 'desses', 'desta', 'destas', 'deste',
    'destes', 'deve', 'devem', 'devendo', 'dever', 'deverá', 'deverão', 'deveria', 'deveriam', 'devia', 'deviam',
    'dez', 'dezoito', 'dia', 'dias', 'diante', 'diz', 'dizem', 'dizer', 'do', 'dois', 'dos', 'doze', 'duas',
    'dúvida', 'e', 'é', 'ela', 'elas', 'ele', 'eles', 'em', 'embora', 'enquanto', 'entre', 'era', 'eram', 'éramos',
    'essa', 'essas', 'esse', 'esses', 'esta', 'está', 'estamos', 'estão', 'estar', 'estas', 'estava', 'estavam',
    'estávamos', 'este', 'estes', 'esteve', 'estive', 'estivemos', 'estiveram', 'estivesse', 'estivessem', 'estou',
    'eu', 'exemplo', 'faço', 'falta', 'favor', 'faz', 'fazeis', 'fazem', 'fazemos', 'fazer', 'fazes', 'fazia',
    'faziamos', 'faziam', 'fez', 'fim', 'final', 'foi', 'fomos', 'for', 'fora', 'foram', 'fôramos', 'forem',
    'forma', 'formos', 'fosse', 'fossem', 'fôssemos', 'fui', 'geral', 'grande', 'grandes', 'grupo', 'há', 'haja',
    'hajam', 'hajamos', 'hão', 'havemos', 'haver', 'hei', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos',
    'isso', 'isto', 'já', 'la', 'lá', 'lado', 'lhe', 'lhes', 'lo', 'local', 'logo', 'longe', 'lugar', 'maior',
    'maioria', 'mais', 'mal', 'mas', 'máximo', 'me', 'meio', 'menor', 'menos', 'mês', 'meses', 'mesma', 'mesmas',
    'mesmo', 'mesmos', 'meu', 'meus', 'mil', 'minha', 'minhas', 'momento', 'muito', 'muitos', 'na', 'nada', 'não',
    'naquela', 'naquelas', 'naquele', 'naqueles', 'nas', 'nem', 'nenhum', 'nenhuma', 'nessa', 'nessas', 'nesse',
    'nesses', 'nesta', 'nestas', 'neste', 'nestes', 'ninguém', 'nível', 'no', 'noite', 'nome', 'nos', 'nós', 'nossa',
    'nossas', 'nosso', 'nossos', 'nova', 'novas', 'novo', 'novos', 'num', 'numa', 'número', 'nunca', 'o', 'obra',
    'obrigada', 'obrigado', 'oitava', 'oitavo', 'oito', 'onde', 'ontem', 'onze', 'os', 'ou', 'outra', 'outras',
    'outro', 'outros', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'pequena', 'pequenas', 'pequeno', 'pequenos',
    'perante', 'perto', 'pode', 'pôde', 'podem', 'poder', 'poderia', 'poderiam', 'podia', 'podiam', 'põe', 'põem',
    'ponto', 'pontos', 'por', 'porém', 'porque', 'porquê', 'possição', 'possível', 'possivelmente', 'posso', 'pouca',
    'poucas', 'pouco', 'poucos', 'primeira', 'primeiras', 'primeiro', 'primeiros', 'própria', 'próprias', 'próprio',
    'próprios', 'quais', 'qual', 'qualquer', 'quando', 'quanto', 'quantos', 'quarta', 'quarto', 'quatro', 'que',
    'quem', 'quer', 'quereis', 'querem', 'queremos', 'queres', 'quero', 'questão', 'quinta', 'quinto', 'quinze',
    'sabe', 'sabem', 'saber', 'são', 'se', 'segunda', 'segundo', 'sei', 'seis', 'sem', 'sempre', 'sendo', 'ser',
    'será', 'serão', 'seria', 'seriam', 'sete', 'sétima', 'sétimo', 'seu', 'seus', 'si', 'sido', 'só', 'sob',
    'sobre', 'sua', 'suas', 'tal', 'talvez', 'também', 'tampouco', 'tanta', 'tantas', 'tanto', 'tantos', 'te',
    'tem', 'têm', 'temos', 'tendes', 'tendo', 'tenha', 'tenham', 'tenhamos', 'tenho', 'tens', 'ter', 'terá',
    'terão', 'terceira', 'terceiro', 'teria', 'teriam', 'teve', 'ti', 'tido', 'tinha', 'tinham', 'tínhamos',
    'tive', 'tivemos', 'tiver', 'tivera', 'tivéramos', 'tiveram', 'tiverem', 'tivermos', 'tivesse', 'tivessem',
    'tivéssemos', 'toda', 'todas', 'todavia', 'todo', 'todos', 'trabalho', 'três', 'treze', 'tu', 'tua', 'tuas',
    'tudo', 'última', 'últimas', 'último', 'últimos', 'um', 'uma', 'umas', 'uns', 'vai', 'vais', 'vão', 'vários',
    'vem', 'vêm', 'vendo', 'vens', 'ver', 'vez', 'vezes', 'viagem', 'vindo', 'vinte', 'você', 'vocês', 'vos',
    'vós', 'vossa', 'vossas', 'vosso', 'vossos', 'zero', 'pra', 'pro', 'pras', 'pros', 'ta', 'tá', 'né', 'ne'
}

def get_top_ngrams(texts, n=1, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(PT_STOPWORDS), min_df=5, token_pattern=r'(?u)\b[a-záàâãéèêíïóôõöúçñ]{3,}\b')
    X = vec.fit_transform(texts)
    sum_words = X.sum(axis=0)
    words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

print("Top 15 Unigrams - Fake:")
top_uni_fake = get_top_ngrams(df[df['label'] == 'fake']['text_clean'], n=1, top_k=15)
for w, f in top_uni_fake:
    print(f"  {w}: {f}")

print("\nTop 15 Unigrams - True:")
top_uni_true = get_top_ngrams(df[df['label'] == 'true']['text_clean'], n=1, top_k=15)
for w, f in top_uni_true:
    print(f"  {w}: {f}")

print("\nTop 15 Bigrams - Fake:")
top_bi_fake = get_top_ngrams(df[df['label'] == 'fake']['text_clean'], n=2, top_k=15)
for w, f in top_bi_fake:
    print(f"  {w}: {f}")

print("\nTop 15 Bigrams - True:")
top_bi_true = get_top_ngrams(df[df['label'] == 'true']['text_clean'], n=2, top_k=15)
for w, f in top_bi_true:
    print(f"  {w}: {f}")

print("\nTop 15 Trigrams - Fake:")
top_tri_fake = get_top_ngrams(df[df['label'] == 'fake']['text_clean'], n=3, top_k=15)
for w, f in top_tri_fake:
    print(f"  {w}: {f}")

print("\nTop 15 Trigrams - True:")
top_tri_true = get_top_ngrams(df[df['label'] == 'true']['text_clean'], n=3, top_k=15)
for w, f in top_tri_true:
    print(f"  {w}: {f}")

# Plot: Top Bigrams Fake vs True
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

df_bi_fake = pd.DataFrame(top_bi_fake[:10], columns=['bigram', 'count'])
sns.barplot(data=df_bi_fake, y='bigram', x='count', color='#e74c3c', ax=axes[0])
axes[0].set_title("Top 10 Bigramas em FAKE NEWS", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Frequência", fontsize=11)
axes[0].set_ylabel("")

df_bi_true = pd.DataFrame(top_bi_true[:10], columns=['bigram', 'count'])
sns.barplot(data=df_bi_true, y='bigram', x='count', color='#2ecc71', ax=axes[1])
axes[1].set_title("Top 10 Bigramas em NOTÍCIAS VERDADEIRAS", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Frequência", fontsize=11)
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "top_bigrams_comparison.png"), dpi=200)
plt.close()


# 5. Discriminative N-grams (Log-Odds Ratio with Dirichlet Prior)
print("\n=== 4. Most Discriminative Bigrams (Log-Odds Ratio) ===")
def get_log_odds_ngrams(texts_a, texts_b, n=2, min_count=20, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(PT_STOPWORDS), min_df=min_count, token_pattern=r'(?u)\b[a-záàâãéèêíïóôõöúçñ]{3,}\b')
    vec.fit(pd.concat([texts_a, texts_b]))
    
    counts_a = np.asarray(vec.transform(texts_a).sum(axis=0)).ravel()
    counts_b = np.asarray(vec.transform(texts_b).sum(axis=0)).ravel()
    
    total_a = counts_a.sum()
    total_b = counts_b.sum()
    
    # Dirichlet prior
    prior = (counts_a + counts_b) / (total_a + total_b)
    alpha = 0.1 * prior * (total_a + total_b)
    
    log_odds = np.log((counts_a + alpha) / (total_a + alpha.sum() - counts_a - alpha)) - \
               np.log((counts_b + alpha) / (total_b + alpha.sum() - counts_b - alpha))
    
    variance = (1.0 / (counts_a + alpha)) + (1.0 / (counts_b + alpha))
    z_scores = log_odds / np.sqrt(variance)
    
    feature_names = np.array(vec.get_feature_names_out())
    
    # Top for A (Fake)
    top_a_idx = np.argsort(z_scores)[-top_k:][::-1]
    top_a = list(zip(feature_names[top_a_idx], z_scores[top_a_idx]))
    
    # Top for B (True)
    top_b_idx = np.argsort(z_scores)[:top_k]
    top_b = list(zip(feature_names[top_b_idx], -z_scores[top_b_idx]))
    
    return top_a, top_b

disc_fake, disc_true = get_log_odds_ngrams(df[df['label'] == 'fake']['text_clean'], df[df['label'] == 'true']['text_clean'], n=2, top_k=15)
print("Top 15 Most Discriminative Bigrams for FAKE (Z-score):")
for bg, score in disc_fake:
    print(f"  {bg}: {score:.2f}")

print("\nTop 15 Most Discriminative Bigrams for TRUE (Z-score):")
for bg, score in disc_true:
    print(f"  {bg}: {score:.2f}")

# Plot: Discriminative Bigrams
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

df_disc_fake = pd.DataFrame(disc_fake[:10], columns=['bigram', 'z_score'])
sns.barplot(data=df_disc_fake, y='bigram', x='z_score', color='#c0392b', ax=axes[0])
axes[0].set_title("Bigramas Mais Distintivos de FAKE (Z-Score Log-Odds)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Força da Associação (Z-Score)", fontsize=11)
axes[0].set_ylabel("")

df_disc_true = pd.DataFrame(disc_true[:10], columns=['bigram', 'z_score'])
sns.barplot(data=df_disc_true, y='bigram', x='z_score', color='#27ae60', ax=axes[1])
axes[1].set_title("Bigramas Mais Distintivos de TRUE (Z-Score Log-Odds)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Força da Associação (Z-Score)", fontsize=11)
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "discriminative_bigrams.png"), dpi=200)
plt.close()


# 6. Sentiment & Sensationalism Lexicon Analysis
print("\n=== 5. Sentiment & Sensationalism Stylistic Markers ===")

URGENCY_WORDS = [
    'urgente', 'alerta', 'atenção', 'cuidado', 'bomba', 'perigo', 'urgência', 'repasse',
    'compartilhe', 'divulgue', 'espalhe', 'apaguem', 'viralize', 'escondem', 'revelado',
    'censurado', 'segredo', 'absurdo', 'inacreditável', 'vergonha', 'escândalo', 'criminoso',
    'chocante', 'revoltante', 'mídia cala', 'verdade revelada', 'acabe com isso'
]

# Curated Portuguese Sentiment Lexicon
POS_WORDS = {
    'bom', 'boa', 'ótimo', 'ótima', 'excelente', 'maravilhoso', 'maravilhosa', 'sucesso', 'vitória',
    'parabéns', 'positivo', 'esperança', 'avanço', 'conquista', 'benefício', 'saúde', 'cura',
    'eficaz', 'eficiente', 'alegria', 'feliz', 'seguro', 'segurança', 'recuperação', 'crescimento',
    'melhor', 'melhoria', 'ganho', 'favorável', 'justiça', 'verdade', 'verdadeiro', 'correto'
}

NEG_WORDS = {
    'ruim', 'péssimo', 'péssima', 'horrível', 'terrível', 'fracasso', 'derrota', 'crise',
    'negativo', 'perigo', 'ameaça', 'prejuízo', 'doença', 'morte', 'morrer', 'vírus',
    'letal', 'veneno', 'tristeza', 'medo', 'inseguro', 'fraude', 'crime', 'corrupção',
    'mentira', 'falso', 'farsa', 'golpe', 'ataque', 'violência', 'destruição', 'culpado',
    'caos', 'pânico', 'desastre', 'traição', 'covarde', 'vergonha', 'absurdo'
}

def analyze_style_and_sentiment(row):
    text_c = str(row['text_clean']).lower()
    words = text_c.split()
    n_words = len(words)
    if n_words == 0:
        return pd.Series({
            'urgency_score': 0.0,
            'pos_score': 0.0,
            'neg_score': 0.0,
            'sentiment_polarity': 0.0,
            'subjectivity': 0.0
        })
    
    urg_hits = sum(1 for w in URGENCY_WORDS if w in text_c)
    pos_hits = sum(1 for w in words if w in POS_WORDS)
    neg_hits = sum(1 for w in words if w in NEG_WORDS)
    
    urg_score = (urg_hits / n_words) * 100
    pos_score = (pos_hits / n_words) * 100
    neg_score = (neg_hits / n_words) * 100
    
    polarity = (pos_hits - neg_hits) / (pos_hits + neg_hits + 1e-5)
    subjectivity = ((pos_hits + neg_hits) / n_words) * 100
    
    return pd.Series({
        'urgency_score': urg_score,
        'pos_score': pos_score,
        'neg_score': neg_score,
        'sentiment_polarity': polarity,
        'subjectivity': subjectivity
    })

print("Calculating sentiment and sensationalism scores across dataset...")
style_metrics = df.apply(analyze_style_and_sentiment, axis=1)
df = pd.concat([df, style_metrics], axis=1)

print("\n--- Stylistic & Sentiment Metrics by Class ---")
style_summary = df.groupby('label')[['num_exclamations', 'num_questions', 'num_ellipsis', 'uppercase_word_ratio', 'urgency_score', 'pos_score', 'neg_score', 'sentiment_polarity', 'subjectivity']].mean()
print(style_summary.T.to_string())

print("\n--- Stylistic & Sentiment Metrics by Subset ---")
subset_style = df.groupby('dataset_name')[['uppercase_word_ratio', 'urgency_score', 'num_exclamations', 'sentiment_polarity']].mean()
print(subset_style.to_string())

# Plot: Stylistic & Sentiment Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Uppercase Word Ratio
sns.barplot(data=df, x='label', y='uppercase_word_ratio', hue='label', palette={'fake': '#e74c3c', 'true': '#2ecc71'}, ax=axes[0, 0])
axes[0, 0].set_title("Proporção de Palavras em CAIXA ALTA (Shouting)", fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel("Média da Proporção", fontsize=10)
axes[0, 0].set_xlabel("Classe", fontsize=10)

# 2. Urgency / Sensationalism Score
sns.barplot(data=df, x='label', y='urgency_score', hue='label', palette={'fake': '#e74c3c', 'true': '#2ecc71'}, ax=axes[0, 1])
axes[0, 1].set_title("Densidade de Léxico de Urgência / Sensacionalismo", fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel("Menções por 100 palavras", fontsize=10)
axes[0, 1].set_xlabel("Classe", fontsize=10)

# 3. Sentiment Polarity
sns.kdeplot(data=df, x='sentiment_polarity', hue='label', common_norm=False, palette={'fake': '#e74c3c', 'true': '#2ecc71'}, ax=axes[1, 0], fill=True, alpha=0.3)
axes[1, 0].set_title("Densidade da Polaridade de Sentimento (-1 Negativo a +1 Positivo)", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("Polaridade", fontsize=10)
axes[1, 0].set_ylabel("Densidade", fontsize=10)

# 4. Punctuation Intensity (Exclamations)
sns.barplot(data=df, x='source_type', y='num_exclamations', hue='label', palette={'fake': '#e74c3c', 'true': '#2ecc71'}, ax=axes[1, 1])
axes[1, 1].set_title("Média de Pontos de Exclamação por Tipo de Fonte", fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel("Média de Exclamações (!)", fontsize=10)
axes[1, 1].set_xlabel("Tipo de Fonte", fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "stylistic_and_sentiment_comparison.png"), dpi=200)
plt.close()

print("\nEDA completed successfully! Plots saved in:", plots_dir)
