#!/usr/bin/env python3
"""
Script para calcular el número de parámetros de la red neuronal
y verificar si tenemos suficientes datos para el entrenamiento.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

def calculate_model_parameters():
    """
    Calcula el número de parámetros de la arquitectura definida:
    
    Masking(3) -> LSTM(128, return_sequences=True) -> Dense(64, ReLU) -> Dense(3, linear)
    """
    print("=== CÁLCULO DE PARÁMETROS DE LA RED NEURONAL ===\n")
    
    # Parámetros de entrada
    input_features = 3  # dx, dy, dz
    sequence_length = 3600  # 1 hora a 1Hz
    
    print(f"Entrada: [{sequence_length}, {input_features}] = [tiempo, features]")
    print(f"Salida:  [{sequence_length}, {input_features}] = [tiempo, features_filtradas]\n")
    
    total_params = 0
    
    # 1. Capa Masking - No tiene parámetros
    print("1. Masking Layer:")
    print("   - Parámetros: 0 (solo aplica máscara)")
    
    # 2. Capa LSTM
    print("\n2. LSTM Layer (128 unidades):")
    lstm_units = 128
    input_size = input_features
    
    # LSTM tiene 4 gates: input, forget, cell, output
    # Cada gate tiene: W_ih (input-to-hidden) + W_hh (hidden-to-hidden) + bias
    
    # Pesos input-to-hidden: (input_size × lstm_units) × 4 gates
    W_ih = input_size * lstm_units * 4
    
    # Pesos hidden-to-hidden: (lstm_units × lstm_units) × 4 gates  
    W_hh = lstm_units * lstm_units * 4
    
    # Bias: lstm_units × 4 gates
    bias = lstm_units * 4
    
    lstm_params = W_ih + W_hh + bias
    
    print(f"   - Input-to-Hidden: {input_size} × {lstm_units} × 4 = {W_ih:,}")
    print(f"   - Hidden-to-Hidden: {lstm_units} × {lstm_units} × 4 = {W_hh:,}")
    print(f"   - Bias: {lstm_units} × 4 = {bias:,}")
    print(f"   - Total LSTM: {lstm_params:,} parámetros")
    
    total_params += lstm_params
    
    # 3. Primera capa Dense (64 neuronas)
    print("\n3. Dense Layer (64 neuronas, ReLU):")
    dense1_input = lstm_units  # Sale del LSTM
    dense1_output = 64
    
    dense1_weights = dense1_input * dense1_output
    dense1_bias = dense1_output
    dense1_params = dense1_weights + dense1_bias
    
    print(f"   - Pesos: {dense1_input} × {dense1_output} = {dense1_weights:,}")
    print(f"   - Bias: {dense1_output}")
    print(f"   - Total Dense1: {dense1_params:,} parámetros")
    
    total_params += dense1_params
    
    # 4. Capa Dropout - No tiene parámetros
    print("\n4. Dropout Layer:")
    print("   - Parámetros: 0 (solo aplica regularización)")
    
    # 5. Segunda capa Dense (salida)
    print("\n5. Output Dense Layer (3 neuronas, linear):")
    dense2_input = dense1_output
    dense2_output = input_features  # dx, dy, dz
    
    dense2_weights = dense2_input * dense2_output
    dense2_bias = dense2_output
    dense2_params = dense2_weights + dense2_bias
    
    print(f"   - Pesos: {dense2_input} × {dense2_output} = {dense2_weights:,}")
    print(f"   - Bias: {dense2_output}")
    print(f"   - Total Dense2: {dense2_params:,} parámetros")
    
    total_params += dense2_params
    
    print(f"\n" + "="*50)
    print(f"TOTAL DE PARÁMETROS: {total_params:,}")
    print(f"="*50)
    
    return total_params

def analyze_available_data():
    """Analiza la cantidad de datos disponibles para entrenamiento."""
    print("\n=== ANÁLISIS DE DATOS DISPONIBLES ===\n")
    
    # Verificar si existe el manifest
    manifest_path = Path("data/input/manifest.csv")
    if not manifest_path.exists():
        print("❌ No se encontró data/input/manifest.csv")
        print("   Ejecuta primero los scripts 1-5 de preprocesamiento")
        return None
    
    # Cargar manifest
    manifest = pd.read_csv(manifest_path)
    
    print(f"📊 ESTADÍSTICAS DEL DATASET:")
    print(f"   - Total ventanas: {len(manifest):,}")
    print(f"   - Pasadas únicas: {manifest['pasada'].nunique()}")
    print(f"   - Grabaciones únicas: {manifest['grabacion'].nunique()}")
    
    # Analizar por pasada
    print(f"\n📈 DISTRIBUCIÓN POR PASADA:")
    pasada_counts = manifest['pasada'].value_counts().sort_index()
    for pasada, count in pasada_counts.items():
        grabaciones = manifest[manifest['pasada'] == pasada]['grabacion'].nunique()
        print(f"   - Pasada {pasada}: {count:,} ventanas ({grabaciones} grabaciones)")
    
    # Calcular familias reagrupadas
    print(f"\n👥 FAMILIAS REAGRUPADAS:")
    family_groups = {}
    for pasada in manifest['pasada'].unique():
        pasada_str = str(pasada)
        base_num = ''.join(filter(str.isdigit, pasada_str))
        if base_num not in family_groups:
            family_groups[base_num] = []
        family_groups[base_num].append(pasada)
    
    for family_base, pasadas in sorted(family_groups.items()):
        if len(pasadas) > 1:
            pasadas.sort()
            total_ventanas = len(manifest[manifest['pasada'].isin(pasadas)])
            print(f"   - Familia {family_base}: {pasadas} → {total_ventanas:,} ventanas")
    
    print(f"\n   - Total familias: {len(family_groups)}")
    
    # Calcular puntos de datos totales
    total_windows = len(manifest)
    points_per_window = 3600  # Según el plan
    features_per_point = 3    # dx, dy, dz
    
    total_points = total_windows * points_per_window
    total_features = total_points * features_per_point
    
    print(f"\n🎯 VOLUMEN DE DATOS:")
    print(f"   - Ventanas: {total_windows:,}")
    print(f"   - Puntos por ventana: {points_per_window:,}")
    print(f"   - Features por punto: {features_per_point}")
    print(f"   - Total puntos: {total_points:,}")
    print(f"   - Total features: {total_features:,}")
    
    return {
        'total_windows': total_windows,
        'total_points': total_points,
        'total_features': total_features,
        'families': len(family_groups),
        'pasadas': manifest['pasada'].nunique(),
        'grabaciones': manifest['grabacion'].nunique()
    }

def evaluate_data_sufficiency(total_params, data_stats):
    """Evalúa si tenemos suficientes datos para el número de parámetros."""
    print("\n=== EVALUACIÓN: ¿SUFICIENTES DATOS? ===\n")
    
    if data_stats is None:
        print("❌ No se pudo analizar los datos")
        return
    
    total_features = data_stats['total_features']
    total_params = total_params
    
    # Reglas empíricas comunes en ML
    ratio_features_params = total_features / total_params
    
    print(f"📊 RATIOS IMPORTANTES:")
    print(f"   - Total parámetros: {total_params:,}")
    print(f"   - Total features: {total_features:,}")
    print(f"   - Ratio features/parámetros: {ratio_features_params:.1f}")
    
    print(f"\n🎯 EVALUACIÓN SEGÚN REGLAS EMPÍRICAS:")
    
    # Regla 1: Al menos 10 datos por parámetro
    min_features_10x = total_params * 10
    print(f"   📐 Regla 10x: Necesitas ≥{min_features_10x:,} features")
    if total_features >= min_features_10x:
        print(f"      ✅ CUMPLE (tienes {total_features:,})")
    else:
        print(f"      ❌ NO CUMPLE (tienes {total_features:,})")
    
    # Regla 2: Al menos 100 datos por parámetro (regla conservadora)
    min_features_100x = total_params * 100
    print(f"   📐 Regla 100x: Necesitas ≥{min_features_100x:,} features")
    if total_features >= min_features_100x:
        print(f"      ✅ CUMPLE (tienes {total_features:,})")
    else:
        print(f"      ❌ NO CUMPLE (tienes {total_features:,})")
    
    # Evaluación específica para secuencias temporales
    windows = data_stats['total_windows']
    families = data_stats['families']
    
    print(f"\n🔄 EVALUACIÓN PARA LOFO (Leave-One-Family-Out):")
    windows_per_family = windows / families
    print(f"   - Ventanas promedio por familia: {windows_per_family:.1f}")
    print(f"   - En cada ronda LOFO: ~{windows * (families-1)/families:.0f} ventanas para train")
    
    if windows_per_family >= 10:
        print(f"      ✅ Suficientes ventanas por familia para LOFO válido")
    else:
        print(f"      ⚠️ Pocas ventanas por familia - LOFO puede ser inestable")
    
    # Recomendaciones finales
    print(f"\n💡 RECOMENDACIONES:")
    
    if ratio_features_params >= 100:
        print("   ✅ Excelente: Tienes suficientes datos para entrenar")
        print("   ✅ La red no debería sufrir overfitting")
        print("   ✅ LOFO será estadísticamente válido")
    elif ratio_features_params >= 10:
        print("   ⚠️ Aceptable: Datos suficientes pero con precauciones")
        print("   ⚠️ Usa técnicas de regularización (dropout, early stopping)")
        print("   ⚠️ Monitorea overfitting durante entrenamiento")
    else:
        print("   ❌ Insuficiente: Alto riesgo de overfitting")
        print("   ❌ Considera reducir el tamaño de la red")
        print("   ❌ O conseguir más datos")
    
    print(f"\n🚀 OPTIMIZACIONES SUGERIDAS:")
    
    if ratio_features_params < 50:
        print("   • Reducir LSTM de 128 a 64 unidades")
        print("   • Reducir Dense de 64 a 32 neuronas")
        print("   • Aumentar dropout (0.3-0.4)")
    
    print("   • Usar BatchNormalization entre capas")
    print("   • Implementar EarlyStopping agresivo")
    print("   • Usar data augmentation si es posible")

def main():
    """Función principal."""
    print("🧮 CALCULADORA DE PARÁMETROS Y ANÁLISIS DE DATOS\n")
    
    # Calcular parámetros del modelo
    total_params = calculate_model_parameters()
    
    # Analizar datos disponibles
    data_stats = analyze_available_data()
    
    # Evaluar suficiencia
    evaluate_data_sufficiency(total_params, data_stats)
    
    print(f"\n{'='*60}")
    print("ANÁLISIS COMPLETADO")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()