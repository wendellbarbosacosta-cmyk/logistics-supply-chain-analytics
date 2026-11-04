import os
import pytest
import pandas as pd
from src.logistics_pipeline import (
    find_dataset_file,
    load_and_preprocess,
    run_logistics_pipeline
)

DATA_PATH = "data/logistics_supply_chain_dataset.csv"

def test_find_dataset_file():
    resolved = find_dataset_file(DATA_PATH)
    assert os.path.exists(resolved), f"Arquivo não encontrado: {resolved}"

def test_load_and_preprocess():
    df = load_and_preprocess(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1200
    assert "delivery_status" in df.columns
    assert "carrier_name" in df.columns
    assert "distance_km" in df.columns
    assert "promised_sla_hours" in df.columns

def test_run_logistics_pipeline():
    model = run_logistics_pipeline(DATA_PATH)
    assert model is not None
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "feature_importances_")
