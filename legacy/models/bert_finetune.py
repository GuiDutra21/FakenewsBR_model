import os
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

DATASET_PATH = "data/FakenewsBR_sanitized.csv"  # relativo a raiz do repo
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 2

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def apply_entity_masking(text):
    if not isinstance(text, str):
        return ""
    # Stop-words temporais e jargões pt-pt
    text = re.sub(r'\b(feira|nesta|quarta|terça|terca|segunda|quinta|sexta|sabado|domingo|segundo|hoje|ontem|amanha)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(facto|equipa|ecrã|ecra|polígrafo|poligrafo|observador|contacto)\b', '', text, flags=re.IGNORECASE)
    
    # Entity Masking
    text = re.sub(r'\b(lula|bolsonaro|dilma|temer|doria|ciro|haddad)\b', '[POLITICO]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(cloroquina|ivermectina|vacina|coronavac|pfizer|astrazeneca)\b', '[SAUDE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(stf|tse|ministério público|ministerio publico)\b', '[INSTITUICAO]', text, flags=re.IGNORECASE)
    
    return re.sub(r'\s+', ' ', text).strip()

def main():
    print("=== Configurando Hardware ===")
    if torch.is_vulkan_available():
        device = torch.device("vulkan")
        print(f"Device selecionado: {device} (Aceleração Vulkan ativada)")
    else:
        print("Vulkan não está disponível nesta compilação do PyTorch. Usando CPU.")
        device = torch.device("cpu")

    print("\n=== Carregando e Sanitizando Dados ===")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df_binary = df[df['label'].isin(['fake', 'true'])].copy()
    df_binary['target'] = (df_binary['label'] == 'fake').astype(int)
    df_binary = df_binary.dropna(subset=['text_clean'])
    
    # Para PoC rápido e poupar VRAM, pegaremos uma amostra estratificada se a base for muito grande.
    # Mas como a base tem ~39k registros, 2 épocas podem levar 1-2h dependendo da GPU.
    # Vamos manter a base toda já que a GPU foi "forçada ao máximo".
    
    print("Aplicando máscaras de entidades (Entity Masking)...")
    df_binary['text_masked'] = df_binary['text_clean'].apply(apply_entity_masking)
    
    # Splits (70% Train, 15% Val, 15% Test)
    print("\n=== Criando Splits (70% / 15% / 15%) ===")
    X = df_binary['text_masked'].tolist()
    y = df_binary['target'].tolist()
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    print(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")
    
    print("\n=== Carregando Tokenizador e Modelo (BERTimbau) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
    val_dataset = FakeNewsDataset(X_val, y_val, tokenizer)
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    print("\n=== Iniciando Treinamento ===")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        for batch in tqdm(val_loader, desc='Validating'):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            val_preds.extend(predictions.cpu().numpy())
            val_labels.extend(batch['labels'].cpu().numpy())
            
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1 (Fake): {val_f1:.4f}")

    print("\n=== Avaliação no Conjunto de Teste (15%) ===")
    model.eval()
    test_preds, test_labels = [], []
    for batch in tqdm(test_loader, desc='Testing'):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        test_preds.extend(predictions.cpu().numpy())
        test_labels.extend(batch['labels'].cpu().numpy())
        
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    print(f"\nRESULTADOS FINAIS (TESTE): Acurácia: {test_acc:.4f} | F1-Score Falso: {test_f1:.4f}")
    
    # Save the model
    print("Salvando modelo ajustado em ./models/bertimbau_fakenewsbr")
    model.save_pretrained("./models/bertimbau_fakenewsbr")
    tokenizer.save_pretrained("./models/bertimbau_fakenewsbr")

if __name__ == "__main__":
    main()
