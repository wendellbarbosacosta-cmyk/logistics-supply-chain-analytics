#!/usr/bin/env python3
"""
Pipeline de Análise e Previsão de Risco Logístico (OTIF & SLAs)
Autor: Portfólio de Ciência de Dados & Supply Chain
Execução: python src/logistics_pipeline.py --data data/logistics_supply_chain_dataset.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Suporte seguro a UTF-8 no console Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def find_dataset_file(filepath: str) -> str:
    """Resolve o caminho do arquivo de dados em múltiplos locais comuns."""
    if os.path.exists(filepath):
        return filepath
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join("data", filepath),
        os.path.join(script_dir, "..", "data", os.path.basename(filepath)),
        os.path.join(script_dir, "..", filepath),
        os.path.join(script_dir, filepath),
        os.path.join("..", "data", os.path.basename(filepath)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return filepath

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Carrega o dataset e valida os dados de entregas."""
    resolved_path = find_dataset_file(filepath)
    print(f"[1/4] Carregando dados de entregas de '{resolved_path}'...")
    try:
        df = pd.read_csv(resolved_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filepath}' não foi encontrado.")
        sys.exit(1)

    total_deliveries = len(df)
    on_time = (df['delivery_status'] == 'No Prazo').sum()
    otif = (on_time / total_deliveries) * 100
    penalties = df['sla_breach_penalty'].sum()

    print(f"      Total de Cargas Monitoradas: {total_deliveries:,}")
    print(f"      OTIF Global da Frota (On-Time): {otif:.2f}%")
    print(f"      Custo de Penalidades por Quebra de SLA: R$ {penalties:,.2f}")
    print(f"      Lead Time Médio de Entrega: {df['transit_time_hours'].mean():.1f} horas")
    return df

def run_logistics_pipeline(filepath: str):
    df = load_and_preprocess(filepath)

    print("\n[2/4] Scorecard de Performance das Transportadoras:")
    scorecard = df.groupby('carrier_name').agg(
        volume=('delivery_id', 'count'),
        otif=('delivery_status', lambda s: (s == 'No Prazo').mean() * 100),
        lead_time_horas=('transit_time_hours', 'mean'),
        multas_totais=('sla_breach_penalty', 'sum')
    ).round(2).sort_values(by='otif', ascending=False)
    print(scorecard.to_string())

    features = [
        'distance_km', 'package_weight_kg', 'cargo_value',
        'promised_sla_hours', 'weather_adverse', 'traffic_congestion'
    ]
    X = df[features]
    y = (df['delivery_status'] == 'Atrasado').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Padronização de features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[3/4] Treinando modelos preditivos (Baseline vs Random Forest)...")

    # 1. Regressão Logística
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = lr_model.predict(X_test_scaled)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # 2. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=120, max_depth=6, random_state=42, class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_model.predict(X_test)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print("\n[4/4] Comparativo de Performance no Conjunto de Teste:")
    print("-" * 65)
    print(f"{'Métrica':<25} | {'Regressão Logística':<18} | {'Random Forest':<18}")
    print("-" * 65)
    print(f"{'ROC-AUC Score':<25} | {auc_lr:<18.4f} | {auc_rf:<18.4f}")
    print(f"{'Acurácia Global':<25} | {acc_lr*100:<17.1f}% | {acc_rf*100:<17.1f}%")
    print("-" * 65)

    print("\nRelatório de Classificação (Random Forest):\n")
    print(classification_report(y_test, y_pred_rf, target_names=['No Prazo', 'Atrasado'], zero_division=0))

    print("Importância das Variáveis no Risco de Atraso:")
    for feat, imp in sorted(zip(features, rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  * {feat:<24}: {imp*100:.1f}%")

    print("\n[OK] Pipeline de Logística e Supply Chain finalizado com sucesso!")
    return rf_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Risco Logístico e SLAs")
    default_csv = "data/logistics_supply_chain_dataset.csv"
    if not os.path.exists(default_csv) and os.path.exists("logistics_supply_chain_dataset.csv"):
        default_csv = "logistics_supply_chain_dataset.csv"
    parser.add_argument('--data', type=str, default=default_csv, help="Caminho do arquivo CSV de entregas")
    args = parser.parse_args()
    run_logistics_pipeline(args.data)
