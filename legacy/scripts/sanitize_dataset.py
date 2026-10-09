import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sanitize():
    input_path = str(ROOT / "data" / "FakenewsBR_factchecked.csv")
    output_sanitized_csv = str(ROOT / "data" / "FakenewsBR_sanitized.csv")
    output_conflicts_csv = str(ROOT / "data" / "conflicting_labels_audit.csv")
    
    print("1. Loading raw dataset...")
    df = pd.read_csv(input_path, low_memory=False)
    total_initial = len(df)
    print(f"Total initial rows: {total_initial}")

    # 2. Identify conflicting labels
    print("2. Detecting conflicting labels...")
    label_counts_per_text = df.groupby('text_clean')['label'].nunique()
    conflict_texts = set(label_counts_per_text[label_counts_per_text > 1].index)
    
    df_conflicts = df[df['text_clean'].isin(conflict_texts)].copy()
    print(f"Unique texts with conflicting labels: {len(conflict_texts)}")
    print(f"Total rows involved in conflicts: {len(df_conflicts)}")
    
    # Enrich conflict audit with conflict details
    conflict_summary = df_conflicts.groupby('text_clean').agg(
        subsets_involved=('dataset_name', lambda x: list(x)),
        labels_involved=('label', lambda x: list(x)),
        row_count=('rid', 'count')
    ).reset_index()
    
    df_conflicts_audit = df_conflicts.merge(conflict_summary, on='text_clean', how='left')
    df_conflicts_audit.to_csv(output_conflicts_csv, index=False, encoding='utf-8')
    print(f"Saved conflict audit to: {output_conflicts_csv}")

    # 3. Filter out conflicts
    df_clean = df[~df['text_clean'].isin(conflict_texts)].copy()
    print(f"Rows remaining after conflict removal: {len(df_clean)}")

    # 4. Deduplication strategy:
    # We prioritize rows that have:
    # a) url_review (fact-check link)
    # b) factcheck_rating
    # c) date_iso
    # d) non-raw dataset over raw dataset
    def compute_priority(row):
        score = 0
        if pd.notna(row['url_review']) and str(row['url_review']).strip() != '':
            score += 4
        if pd.notna(row['factcheck_rating']) and str(row['factcheck_rating']).strip() != '':
            score += 3
        if pd.notna(row['factcheck_claimant']) and str(row['factcheck_claimant']).strip() != '':
            score += 2
        if pd.notna(row['date_iso']) and str(row['date_iso']).strip() != '':
            score += 2
        if not str(row['dataset_name']).endswith('_raw'):
            score += 1
        return score

    print("4. Calculating row quality / metadata richness score for deduplication...")
    df_clean['priority_score'] = df_clean.apply(compute_priority, axis=1)

    # Sort descending by priority_score, then keep the first occurrence for each text_clean
    df_dedup = df_clean.sort_values(by='priority_score', ascending=False).drop_duplicates(subset=['text_clean']).copy()
    df_dedup = df_dedup.drop(columns=['priority_score'])
    
    # 5. Fix known dataset anomalies:
    # Fake.br was tagged as 'tweets' in source_type, but it's news articles from Fake.br corpus
    mask_fakebr = df_dedup['dataset_name'].isin(['Fake.br', 'Fake.br_raw']) & (df_dedup['source_type'] == 'tweets')
    df_dedup.loc[mask_fakebr, 'source_type'] = 'news'

    # 6. Compute text statistics & stylistic markers
    print("5. Computing stylistic & text metrics...")
    
    # Char & word lengths on text_clean
    df_dedup['char_len'] = df_dedup['text_clean'].fillna('').astype(str).apply(len)
    df_dedup['word_len'] = df_dedup['text_clean'].fillna('').astype(str).apply(lambda x: len(x.split()))
    
    # Punctuation markers on raw text
    def count_pattern(text, pattern):
        if not isinstance(text, str):
            return 0
        return len(re.findall(pattern, text))
    
    df_dedup['num_exclamations'] = df_dedup['text'].apply(lambda t: count_pattern(t, r'!'))
    df_dedup['num_questions'] = df_dedup['text'].apply(lambda t: count_pattern(t, r'\?'))
    df_dedup['num_ellipsis'] = df_dedup['text'].apply(lambda t: count_pattern(t, r'\.{3,}|\u2026'))
    
    # Uppercase words ratio (words with 3+ uppercase letters / total words)
    def uppercase_ratio(text):
        if not isinstance(text, str) or not text:
            return 0.0
        words = re.findall(r'\b[A-Za-zÀ-ÖØ-öø-ÿ]+\b', text)
        if not words:
            return 0.0
        upper_words = sum(1 for w in words if len(w) >= 3 and w.isupper())
        return upper_words / len(words)
        
    df_dedup['uppercase_word_ratio'] = df_dedup['text'].apply(uppercase_ratio)

    # 7. Save sanitized dataset
    print(f"6. Saving sanitized dataset ({len(df_dedup)} rows)...")
    df_dedup.to_csv(output_sanitized_csv, index=False, encoding='utf-8')
    
    print("\n=== SANITIZATION SUMMARY ===")
    print(f"Initial rows: {total_initial}")
    print(f"Conflicting rows isolated: {len(df_conflicts)} (from {len(conflict_texts)} unique texts)")
    print(f"Duplicates removed: {total_initial - len(df_conflicts) - len(df_dedup)}")
    print(f"Final sanitized rows: {len(df_dedup)}")
    print("\nSanitized Label Distribution:")
    print(df_dedup['label'].value_counts(normalize=False))
    print(df_dedup['label'].value_counts(normalize=True) * 100)
    print("\nSanitized Subsets Distribution:")
    print(pd.crosstab(df_dedup['dataset_name'], df_dedup['label'], margins=True))
    print("\nSanitized Source Types:")
    print(pd.crosstab(df_dedup['source_type'], df_dedup['label'], margins=True))

if __name__ == '__main__':
    sanitize()
