#!/usr/bin/env python3
"""Script de diagnóstico para identificar dónde falla el entrenamiento."""

import sys
import traceback

def test_step(step_name, func):
    """Ejecuta un paso y captura errores."""
    print(f"🔍 {step_name}... ", end="", flush=True)
    try:
        result = func()
        print("✅ OK")
        return result
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return None

def main():
    print("=== DIAGNÓSTICO DEL SCRIPT DE ENTRENAMIENTO ===\n")
    
    # Paso 1: Importaciones básicas
    def test_imports():
        import numpy as np
        import pandas as pd
        import json
        from pathlib import Path
        return True
    
    if not test_step("Importaciones básicas", test_imports):
        return
    
    # Paso 2: Importaciones de TensorFlow
    def test_tensorflow():
        import tensorflow as tf
        print(f"\n  TensorFlow version: {tf.__version__}")
        return True
    
    if not test_step("TensorFlow", test_tensorflow):
        return
    
    # Paso 3: Carga de archivos
    def test_files():
        import pandas as pd
        import json
        from pathlib import Path
        
        # Verificar archivos necesarios
        manifest_path = Path("data/input/manifest.csv")
        norm_stats_path = Path("data/input/norm_stats.json")
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"No existe: {manifest_path}")
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"No existe: {norm_stats_path}")
            
        # Cargar archivos
        manifest = pd.read_csv(manifest_path)
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
            
        print(f"\n  Manifest: {manifest.shape}")
        print(f"  Norm stats keys: {list(norm_stats.keys())}")
        return manifest, norm_stats
    
    result = test_step("Carga de archivos", test_files)
    if not result:
        return
    manifest, norm_stats = result
    
    # Paso 4: Crear familias
    def test_families():
        pasadas_unicas = sorted(manifest['pasada'].unique())
        family_groups = {}
        
        for pasada in pasadas_unicas:
            pasada_str = str(pasada)
            base_num = ''.join(filter(str.isdigit, pasada_str))
            
            if base_num not in family_groups:
                family_groups[base_num] = []
            family_groups[base_num].append(pasada)
        
        print(f"\n  Pasadas únicas: {len(pasadas_unicas)}")
        print(f"  Familias: {len(family_groups)}")
        return family_groups
    
    family_groups = test_step("Creación de familias", test_families)
    if not family_groups:
        return
    
    # Paso 5: Cargar una ventana de prueba
    def test_load_window():
        from pathlib import Path
        import pandas as pd
        
        # Tomar primera fila del manifest
        row = manifest.iloc[0]
        print(f"\n  Probando ventana: {row['slice_path']}")
        
        # Cargar archivos
        slice_path = Path(row['slice_path'])
        label_path = Path(row['label_path'])
        mask_path = Path(row['mask_path'])
        
        input_data = pd.read_csv(slice_path)
        label_data = pd.read_csv(label_path)
        mask_data = pd.read_csv(mask_path)
        
        print(f"  Input shape: {input_data.shape}")
        print(f"  Label shape: {label_data.shape}")
        print(f"  Mask shape: {mask_data.shape}")
        
        # Verificar columnas
        print(f"  Input cols: {list(input_data.columns)}")
        print(f"  Label cols: {list(label_data.columns)}")
        print(f"  Mask cols: {list(mask_data.columns)}")
        
        return True
    
    if not test_step("Carga de ventana de prueba", test_load_window):
        return
    
    # Paso 6: Crear modelo simple
    def test_model():
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Masking
        
        model = Sequential([
            Masking(mask_value=0.0, input_shape=(3600, 3)),
            LSTM(128, return_sequences=True),
            Dense(3)
        ])
        
        print(f"\n  Modelo creado: {len(model.layers)} capas")
        return model
    
    model = test_step("Creación de modelo", test_model)
    if not model:
        return
    
    print("\n✅ TODOS LOS PASOS COMPLETADOS EXITOSAMENTE")
    print("El problema debe estar en la lógica del script principal.")

if __name__ == "__main__":
    main()