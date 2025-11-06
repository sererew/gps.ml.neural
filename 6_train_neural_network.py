#!/usr/bin/env python3
"""
Script para entrenar la red neuronal de corrección de tracks GPS.

La red toma como entrada secuencias de deltas (dx, dy, dz) de grabaciones ruidosas
y aprende a predecir los deltas del track patrón limpio correspondiente.

Arquitectura:
- LSTM(128) con return_sequences=True
- Dense(64) con activación ReLU  
- Output(3) con activación lineal para (dx, dy, dz)

Entrada: [batch, time, features] donde features = [dx, dy, dz]
Salida: [batch, time, features] donde features = [dx_pred, dy_pred, dz_pred]
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import MeanAbsoluteError, Huber
import tensorflow.keras.backend as K

# Configuración de GPU si está disponible
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU disponible: {physical_devices[0]}")
else:
    print("Entrenando en CPU")

class GPSTrackDataset:
    """Dataset para cargar y procesar datos de tracks GPS."""
    
    def __init__(self, data_dir="data/input"):
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / "manifest.csv"
        self.norm_stats_path = self.data_dir / "norm_stats.json"
        
        # Cargar metadatos
        self.manifest = pd.read_csv(self.manifest_path)
        with open(self.norm_stats_path, 'r') as f:
            self.norm_stats = json.load(f)

        # Normalizar rutas en manifest para que funcionen en Windows y Linux
        self._normalize_manifest_paths()
            
        # Crear mapa de familias reagrupadas
        self.family_groups = self._create_family_groups()
        
        print(f"Dataset cargado: {len(self.manifest)} ventanas")
        print(f"Pasadas únicas: {sorted(self.manifest['pasada'].unique())}")
        print(f"Familias reagrupadas: {len(self.family_groups)} familias")
        for family_base, pasadas in self.family_groups.items():
            total_ventanas = len(self.manifest[self.manifest['pasada'].isin(pasadas)])
            total_grabaciones = self.manifest[self.manifest['pasada'].isin(pasadas)]['grabacion'].nunique()
            if len(pasadas) > 1:
                print(f"  Familia {family_base}: {pasadas} → {total_grabaciones} grab, {total_ventanas} ventanas")
    
    def _create_family_groups(self):
        """Reagrupa pasadas por su número base (4, 4b, 4c, 4d → familia '4')."""
        pasadas_unicas = sorted(self.manifest['pasada'].unique())
        family_groups = {}
        
        for pasada in pasadas_unicas:
            pasada_str = str(pasada)
            
            # Extraer número base (parte numérica)
            base_num = ''.join(filter(str.isdigit, pasada_str))
            
            if base_num not in family_groups:
                family_groups[base_num] = []
            family_groups[base_num].append(pasada)
        
        # Ordenar pasadas dentro de cada familia
        for family_base in family_groups:
            family_groups[family_base] = sorted(family_groups[family_base])
        
        return family_groups

    def _normalize_manifest_paths(self):
        """Replace backslashes with forward slashes in path columns so Path(...) works cross-platform.
        This keeps relative paths as-is (with normalized separators) so they resolve relative to the
        current working directory when the script runs.
        """
        path_cols = ['slice_path', 'label_path', 'mask_path']
        for col in path_cols:
            if col in self.manifest.columns:
                # Convert NaNs to empty strings to avoid errors, then normalize separators
                self.manifest[col] = self.manifest[col].fillna('').astype(str).apply(lambda p: p.replace('\\', '/'))

    def get_family_base_list(self):
        """Retorna lista de familias base para LOFO."""
        return sorted(self.family_groups.keys())
    
    def get_pasadas_for_family(self, family_base):
        """Retorna todas las pasadas (incluyendo derivadas) de una familia base."""
        return self.family_groups.get(str(family_base), [])

    def load_window_data(self, row):
        """Carga datos de una ventana específica."""
        # Normalizar separadores y expandir ~ si aparece. Dejar rutas relativas como relativas
        def _to_path(p):
            if pd.isna(p) or str(p) == '':
                return Path(p)
            pstr = str(p).replace('\\', '/')
            return Path(pstr).expanduser()

        slice_path = _to_path(row['slice_path'])
        input_data = pd.read_csv(slice_path)
        
        # Cargar datos de etiquetas (patrón limpio)
        label_path = _to_path(row['label_path'])
        label_data = pd.read_csv(label_path)
        
        # Cargar máscara
        mask_path = _to_path(row['mask_path'])
        mask_data = pd.read_csv(mask_path)
        
        # Extraer características (dx, dy, dz) - CORRECTO según tus datos
        input_features = input_data[['dx', 'dy', 'dz']].values
        label_features = label_data[['dx', 'dy', 'dz']].values
        mask_values = mask_data['mask'].values  # Usar la columna 'mask' directamente
        
        return input_features, label_features, mask_values
    
    def create_dataset(self, test_pasadas=None, max_samples=None):
        """
        Crea dataset de entrenamiento/validación/test.
        
        Args:
            test_pasadas: Lista de pasadas para test (LOFO). Si None, split aleatorio.
            max_samples: Máximo número de muestras a cargar (para debugging)
        """
        if test_pasadas is not None:
            # Split LOFO (Leave-One-Family-Out)
            train_mask = ~self.manifest['pasada'].isin(test_pasadas)
            train_manifest = self.manifest[train_mask]
            test_manifest = self.manifest[~train_mask]
        else:
            # Split aleatorio 80/20
            train_manifest, test_manifest = train_test_split(
                self.manifest, test_size=0.2, random_state=42, 
                stratify=self.manifest['pasada']
            )
        
        print(f"Train: {len(train_manifest)} ventanas")
        print(f"Test: {len(test_manifest)} ventanas")
        
        # Limitar muestras si se especifica (para debugging)
        if max_samples:
            train_manifest = train_manifest.head(max_samples)
            test_manifest = test_manifest.head(max_samples // 4)
            print(f"Limitado a {len(train_manifest)} train, {len(test_manifest)} test")
        
        # Cargar datos de entrenamiento
        X_train, y_train, masks_train = self._load_data_batch(train_manifest)
        X_test, y_test, masks_test = self._load_data_batch(test_manifest)
        
        # Verificar que tenemos datos suficientes
        if len(X_train) == 0:
            raise ValueError(f"No se pudieron cargar datos de entrenamiento. Revise las rutas en manifest.csv")
        
        if len(X_test) == 0:
            raise ValueError(f"No se pudieron cargar datos de test. Revise las rutas en manifest.csv")
        
        # Split train/validation - solo si tenemos suficientes datos
        if len(X_train) < 5:
            print(f"⚠️ Advertencia: Muy pocos datos de entrenamiento ({len(X_train)}). Usando los mismos datos para validación.")
            X_val, y_val, masks_val = X_train.copy(), y_train.copy(), masks_train.copy()
        else:
            X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(
                X_train, y_train, masks_train, test_size=0.2, random_state=42
            )
        
        print(f"Final - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test)
    
    def _load_data_batch(self, manifest_subset):
        """Carga un lote de datos del manifest."""
        X_list, y_list, masks_list = [], [], []
        
        for idx, row in manifest_subset.iterrows():
            try:
                input_features, label_features, mask_values = self.load_window_data(row)
                X_list.append(input_features)
                y_list.append(label_features)
                masks_list.append(mask_values)
            except Exception as e:
                print(f"Error cargando ventana {row['slice_path']}: {e}")
                continue
        
        # Convertir a arrays numpy
        X = np.array(X_list)
        y = np.array(y_list) 
        masks = np.array(masks_list)
        
        return X, y, masks

def masked_mae_loss(y_true, y_pred):
    """
    Mean Absolute Error que ignora valores enmascarados.
    Asume que los valores enmascarados son 0 en y_true.
    """
    # Crear máscara: verdadero donde NO hay padding (no todos los valores son 0)
    mask = tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True) > 1e-7
    mask = tf.cast(mask, tf.float32)
    
    # Calcular MAE solo en posiciones válidas
    mae = tf.abs(y_true - y_pred)
    masked_mae = mae * mask
    
    # Promedio sobre posiciones válidas
    sum_mae = tf.reduce_sum(masked_mae)
    sum_mask = tf.reduce_sum(mask) 
    
    return sum_mae / (sum_mask + 1e-7)  # Evitar división por 0

def create_model(sequence_length=3600, n_features=3):
    """Crea el modelo de red neuronal."""
    model = Sequential([
        # Capa de enmascaramiento para ignorar padding
        Masking(mask_value=0.0, input_shape=(sequence_length, n_features)),
        
        # LSTM con 128 unidades, devuelve secuencias completas
        # LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
        LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.0),
        
        # Capa densa con 64 neuronas y ReLU
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        # Capa de salida con 3 neuronas (dx, dy, dz) y activación lineal
        Dense(n_features, activation='linear')
    ])
    
    return model

def plot_training_history(history, save_path="training_history.png"):
    """Grafica la historia de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Learning rate si está disponible
    if 'lr' in history.history:
        ax2.plot(history.history['lr'], label='Learning Rate')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('LR')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado en: {save_path}")

def evaluate_model(model, X_test, y_test, masks_test, norm_stats=None):
    """Evalúa el modelo y calcula métricas en valores normalizados y metros reales."""
    print("\n=== EVALUACIÓN DEL MODELO ===")
    
    # Predicciones
    y_pred = model.predict(X_test, batch_size=32, verbose=1)
    
    # Aplicar máscaras para calcular métricas solo en posiciones válidas
    valid_positions = masks_test.astype(bool)
    
    # MAE en valores normalizados
    mae_dx_norm = np.mean(np.abs(y_test[valid_positions][:, 0] - y_pred[valid_positions][:, 0]))
    mae_dy_norm = np.mean(np.abs(y_test[valid_positions][:, 1] - y_pred[valid_positions][:, 1]))  
    mae_dz_norm = np.mean(np.abs(y_test[valid_positions][:, 2] - y_pred[valid_positions][:, 2]))
    mae_total_norm = np.mean([mae_dx_norm, mae_dy_norm, mae_dz_norm])
    
    print(f"MAE dx (normalizado): {mae_dx_norm:.6f}")
    print(f"MAE dy (normalizado): {mae_dy_norm:.6f}")
    print(f"MAE dz (normalizado): {mae_dz_norm:.6f}")
    print(f"MAE total (normalizado): {mae_total_norm:.6f}")
    
    # Si tenemos estadísticas de normalización, calcular MAE en metros reales
    if norm_stats:
        # Desnormalizar predicciones y valores reales
        y_pred_meters = y_pred.copy()
        y_test_meters = y_test.copy()
        
        for i, feature in enumerate(['dx', 'dy', 'dz']):
            std = norm_stats['std'][feature]
            mean = norm_stats['mean'][feature]
            
            y_pred_meters[..., i] = y_pred[..., i] * std + mean
            y_test_meters[..., i] = y_test[..., i] * std + mean
        
        # MAE en metros reales
        mae_dx_meters = np.mean(np.abs(y_test_meters[valid_positions][:, 0] - y_pred_meters[valid_positions][:, 0]))
        mae_dy_meters = np.mean(np.abs(y_test_meters[valid_positions][:, 1] - y_pred_meters[valid_positions][:, 1]))  
        mae_dz_meters = np.mean(np.abs(y_test_meters[valid_positions][:, 2] - y_pred_meters[valid_positions][:, 2]))
        mae_total_meters = np.mean([mae_dx_meters, mae_dy_meters, mae_dz_meters])
        
        print(f"\nMAE dx (metros): {mae_dx_meters:.4f} m")
        print(f"MAE dy (metros): {mae_dy_meters:.4f} m")
        print(f"MAE dz (metros): {mae_dz_meters:.4f} m")
        print(f"MAE total (metros): {mae_total_meters:.4f} m")
        
        return {
            'mae_dx_norm': mae_dx_norm,
            'mae_dy_norm': mae_dy_norm, 
            'mae_dz_norm': mae_dz_norm,
            'mae_total_norm': mae_total_norm,
            'mae_dx_meters': mae_dx_meters,
            'mae_dy_meters': mae_dy_meters,
            'mae_dz_meters': mae_dz_meters,
            'mae_total_meters': mae_total_meters
        }
    else:
        return {
            'mae_dx': mae_dx_norm,
            'mae_dy': mae_dy_norm, 
            'mae_dz': mae_dz_norm,
            'mae_total': mae_total_norm
        }

def run_lofo_validation(dataset, model_config, max_families=None, max_samples_per_family=None):
    """
    Ejecuta validación Leave-One-Family-Out según el plan del proyecto.
    Usa familias reagrupadas (4,4b,4c,4d como una sola familia).
    
    Returns:
        dict: Resultados de LOFO con métricas por familia y promedio
    """
    print("\n=== VALIDACIÓN LEAVE-ONE-FAMILY-OUT (LOFO) ===")
    
    # Usar familias base reagrupadas en lugar de pasadas individuales
    family_bases = dataset.get_family_base_list()
    lofo_results = {}
    
    # Limitar familias si se especifica
    if max_families is not None:
        family_bases = family_bases[:max_families]
        print(f"⚠️ Limitando a las primeras {max_families} familias: {family_bases}")
    
    for test_family_base in family_bases:
        # Obtener todas las pasadas (incluyendo derivadas) de esta familia
        test_pasadas = dataset.get_pasadas_for_family(test_family_base)
        
        print(f"\n--- LOFO Round: Familia {test_family_base} como TEST ---")
        print(f"  Pasadas incluidas: {test_pasadas}")
        
        # Crear split LOFO: una familia completa para test, resto para train
        (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test) = \
            dataset.create_dataset(test_pasadas=test_pasadas, max_samples=max_samples_per_family)
        
        # Skip si no hay suficientes datos para entrenar
        if X_train.shape[0] < 10:
            print(f"  ⚠️  Saltando familia {test_family_base}: muy pocos datos de entrenamiento ({X_train.shape[0]})")
            continue
            
        # Skip si no hay suficientes datos para test
        if X_test.shape[0] < 3:
            print(f"  ⚠️  Saltando familia {test_family_base}: muy pocos datos de test ({X_test.shape[0]})")
            continue
        
        # Crear modelo para esta ronda
        model = create_model(sequence_length=X_train.shape[1], n_features=X_train.shape[2])
        model.compile(
            optimizer=Adam(learning_rate=model_config['learning_rate']),
            loss=masked_mae_loss,
            metrics=['mae']
        )
        
        # Entrenar modelo
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=model_config['patience'],
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        print(f"  🔄 Entrenando modelo (max {model_config['epochs']} épocas)...")
        history = model.fit(
            X_train, y_train,
            batch_size=model_config['batch_size'],
            epochs=model_config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1  # Cambiar a verbose=1 para ver el progreso
        )
        
        # Mostrar resumen del entrenamiento
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        epochs_trained = len(history.history['loss'])
        
        print(f"  ✅ Entrenamiento completado:")
        print(f"    - Épocas entrenadas: {epochs_trained}/{model_config['epochs']}")
        print(f"    - Loss final train: {final_train_loss:.6f}")
        print(f"    - Loss final val: {final_val_loss:.6f}")
        
        # Evaluar en familia test
        metrics = evaluate_model(model, X_test, y_test, masks_test, norm_stats=dataset.norm_stats)
        lofo_results[test_family_base] = metrics
        
        # Usar clave correcta según si tenemos norm_stats o no
        mae_key = 'mae_total_norm' if 'mae_total_norm' in metrics else 'mae_total'
        print(f"  Familia {test_family_base} - MAE total: {metrics[mae_key]:.6f}")
    
    # Calcular estadísticas LOFO usando las claves correctas
    if not lofo_results:
        raise ValueError("No se pudo completar ninguna ronda de LOFO válida")
    
    first_family_key = list(lofo_results.keys())[0]
    mae_key = 'mae_total_norm' if 'mae_total_norm' in lofo_results[first_family_key] else 'mae_total'
    dx_key = 'mae_dx_norm' if 'mae_dx_norm' in lofo_results[first_family_key] else 'mae_dx'
    dy_key = 'mae_dy_norm' if 'mae_dy_norm' in lofo_results[first_family_key] else 'mae_dy'
    dz_key = 'mae_dz_norm' if 'mae_dz_norm' in lofo_results[first_family_key] else 'mae_dz'
    
    valid_families = list(lofo_results.keys())
    mae_totals = [lofo_results[f][mae_key] for f in valid_families]
    mae_dx_values = [lofo_results[f][dx_key] for f in valid_families]
    mae_dy_values = [lofo_results[f][dy_key] for f in valid_families]
    mae_dz_values = [lofo_results[f][dz_key] for f in valid_families]
    
    lofo_stats = {
        'individual_results': lofo_results,
        'summary': {
            'families_tested': len(valid_families),
            'mae_total_mean': np.mean(mae_totals),
            'mae_total_std': np.std(mae_totals),
            'mae_dx_mean': np.mean(mae_dx_values),
            'mae_dx_std': np.std(mae_dx_values),
            'mae_dy_mean': np.mean(mae_dy_values),
            'mae_dy_std': np.std(mae_dy_values),
            'mae_dz_mean': np.mean(mae_dz_values),
            'mae_dz_std': np.std(mae_dz_values),
        }
    }
    
    print("\n=== RESUMEN LOFO (FAMILIAS REAGRUPADAS) ===")
    print(f"Familias evaluadas: {lofo_stats['summary']['families_tested']}")
    print(f"MAE Total: {lofo_stats['summary']['mae_total_mean']:.6f} ± {lofo_stats['summary']['mae_total_std']:.6f}")
    print(f"MAE dx: {lofo_stats['summary']['mae_dx_mean']:.6f} ± {lofo_stats['summary']['mae_dx_std']:.6f}")
    print(f"MAE dy: {lofo_stats['summary']['mae_dy_mean']:.6f} ± {lofo_stats['summary']['mae_dy_std']:.6f}")
    print(f"MAE dz: {lofo_stats['summary']['mae_dz_mean']:.6f} ± {lofo_stats['summary']['mae_dz_std']:.6f}")
    
    return lofo_stats

def train_final_model(dataset, model_config, max_samples=None):
    """
    Entrena modelo final con todas las familias según el plan del proyecto.
    """
    print("\n=== ENTRENAMIENTO MODELO FINAL (TODAS LAS FAMILIAS) ===")
    
    # Usar todas las familias para entrenamiento
    (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test) = \
        dataset.create_dataset(test_pasadas=None, max_samples=max_samples)  # Split aleatorio para val/test
    
    # Crear modelo final
    model = create_model(sequence_length=X_train.shape[1], n_features=X_train.shape[2])
    model.compile(
        optimizer=Adam(learning_rate=model_config['learning_rate']),
        loss=masked_mae_loss,
        metrics=['mae']
    )
    
    # Callbacks para modelo final
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=model_config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'final_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Entrenamiento final
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=model_config['batch_size'],
        epochs=model_config['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Graficar historia
    plot_training_history(history, "final_model_history.png")
    
    # Evaluación final
    final_metrics = evaluate_model(model, X_test, y_test, masks_test, norm_stats=dataset.norm_stats)
    
    return model, history, final_metrics, training_time

def main():
    """Función principal de entrenamiento."""
    print("=== ENTRENAMIENTO DE RED NEURONAL PARA CORRECCIÓN DE TRACKS GPS ===\n")
    
    try:
        print("🔍 Iniciando función main()...")
        
        # =============================
        # CONFIGURACIÓN DE MODO
        # =============================
        # Cambiar FAST_MODE a True para pruebas rápidas
        FAST_MODE = False  # ← Cambiar aquí entre True (rápido) y False (completo)
        
        print(f"🔍 FAST_MODE configurado: {FAST_MODE}")
        
        if FAST_MODE:
            print("🚀 MODO RÁPIDO ACTIVADO - Solo para pruebas")
            print("   - Máximo 3 familias LOFO")
            print("   - 10 épocas por modelo")
            print("   - 50 ventanas máximo por familia")
            print("   - Tiempo estimado: 5-10 minutos\n")
            
            model_config = {
                'epochs': 10,           # Pocas épocas
                'batch_size': 16,       # Batch más pequeño
                'learning_rate': 1e-3,
                'patience': 5           # Early stopping más agresivo
            }
            MAX_FAMILIES_LOFO = 3      # Solo probar 3 familias
            MAX_SAMPLES_PER_FAMILY = 50 # Limitar datos por familia
            
        else:
            print("⏳ MODO COMPLETO ACTIVADO - Entrenamiento real")
            print("   - Todas las 17 familias LOFO")
            print("   - 100 épocas por modelo") 
            print("   - Todos los datos disponibles")
            print("   - Tiempo estimado: 2-4 horas\n")
            
            model_config = {
                'epochs': 100,
                'batch_size': 64,
                'learning_rate': 1e-3,
                'patience': 15
            }
            MAX_FAMILIES_LOFO = None   # Todas las familias
            MAX_SAMPLES_PER_FAMILY = None # Todos los datos
        
        print(f"🔍 Configuración establecida: {model_config}")
        
        # Cargar dataset
        print("🔍 Intentando cargar dataset...")
        dataset = GPSTrackDataset()
        print("🔍 Dataset cargado exitosamente")
        
        # Ejecutar validación LOFO como dice el plan
        print("🔍 Iniciando validación LOFO...")
        lofo_results = run_lofo_validation(dataset, model_config, MAX_FAMILIES_LOFO, MAX_SAMPLES_PER_FAMILY)
        print("🔍 LOFO completado exitosamente")
        
        # Entrenar modelo final con todas las familias
        print("🔍 Iniciando entrenamiento del modelo final...")
        final_model, history, final_metrics, training_time = train_final_model(dataset, model_config, MAX_SAMPLES_PER_FAMILY)
        print("🔍 Modelo final entrenado exitosamente")
        
        # Guardar todos los resultados
        print("🔍 Guardando resultados...")
        mode_suffix = "_fast" if FAST_MODE else "_complete"
        results_file = f'training_results{mode_suffix}.json'
        
        all_results = {
            'mode': 'fast' if FAST_MODE else 'complete',
            'config': model_config,
            'lofo_validation': lofo_results,
            'final_model': {
                'training_time_minutes': training_time / 60,
                'metrics': final_metrics,
                'config': model_config
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"🔍 Resultados guardados en {results_file}")
        
        print(f"\n=== RESULTADOS FINALES ===")
        print(f"Modo: {'🚀 RÁPIDO' if FAST_MODE else '⏳ COMPLETO'}")
        print(f"LOFO MAE promedio: {lofo_results['summary']['mae_total_mean']:.6f} ± {lofo_results['summary']['mae_total_std']:.6f}")
        if 'mae_total_norm' in final_metrics:
            print(f"Modelo final MAE: {final_metrics['mae_total_norm']:.6f}")
        else:
            print(f"Modelo final MAE: {final_metrics['mae_total']:.6f}")
        print(f"Tiempo total: {training_time/60:.2f} minutos")
        print(f"Resultados guardados en: {results_file}")
        
        if FAST_MODE:
            print(f"\n💡 Para entrenamiento completo, cambia FAST_MODE = False en línea ~515")
            
        print("🔍 Script completado exitosamente")
        
    except Exception as e:
        print(f"❌ ERROR en main(): {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("🔍 Script iniciado - verificando punto de entrada...")
    try:
        print("🔍 Llamando a main()...")
        main()
        print("🔍 main() completado exitosamente")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en el script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)