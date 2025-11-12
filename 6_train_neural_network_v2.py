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

Usa dataset pre-dividido por sets (train/val/test) sin data leakage.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
import argparse
from pathlib import Path
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
    """Dataset para cargar y procesar datos de tracks GPS pre-divididos por sets."""
    
    def __init__(self, data_dir="data/input"):
        self.data_dir = Path(data_dir)
        
        # Rutas de manifests por set y estadísticas de normalización
        self.train_manifest_path = self.data_dir / "train" / "manifest_train.csv"
        self.val_manifest_path = self.data_dir / "val" / "manifest_val.csv"
        self.test_manifest_path = self.data_dir / "test" / "manifest_test.csv"
        self.norm_stats_path = self.data_dir / "norm_stats_train.json"
        
        # Verificar que existen los archivos necesarios
        required_files = [
            self.train_manifest_path,
            self.val_manifest_path, 
            self.test_manifest_path,
            self.norm_stats_path
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"Archivos faltantes: {missing_files}")
        
        # Cargar estadísticas de normalización
        with open(self.norm_stats_path, 'r') as f:
            self.norm_stats = json.load(f)
        
        print(f"Dataset pre-dividido cargado desde {data_dir}")
        print(f"  - Train: {self.train_manifest_path}")
        print(f"  - Val: {self.val_manifest_path}")
        print(f"  - Test: {self.test_manifest_path}")
        print(f"  - Stats: {self.norm_stats_path}")

    def _normalize_manifest_paths(self, manifest_df):
        """Normaliza separadores en rutas para compatibilidad cross-platform."""
        path_cols = ['slice_path', 'label_path', 'mask_path']
        for col in path_cols:
            if col in manifest_df.columns:
                manifest_df[col] = manifest_df[col].fillna('').astype(str).apply(lambda p: p.replace('\\', '/'))

    def load_by_sets(self):
        """
        Carga dataset usando manifests pre-divididos por sets.
        
        Returns:
            tuple: (train_data, val_data, test_data) donde cada elemento es (X, y, masks)
        """
        # Cargar manifests por set
        train_manifest = pd.read_csv(self.train_manifest_path)
        val_manifest = pd.read_csv(self.val_manifest_path)
        test_manifest = pd.read_csv(self.test_manifest_path)
        
        # Normalizar rutas en todos los manifests
        self._normalize_manifest_paths(train_manifest)
        self._normalize_manifest_paths(val_manifest)
        self._normalize_manifest_paths(test_manifest)
        
        print(f"Cargando datos por sets:")
        print(f"  Train: {len(train_manifest)} ventanas")
        print(f"  Val: {len(val_manifest)} ventanas")
        print(f"  Test: {len(test_manifest)} ventanas")
        
        # Cargar datos de cada set
        X_train, y_train, masks_train = self._load_data_batch(train_manifest)
        X_val, y_val, masks_val = self._load_data_batch(val_manifest)
        X_test, y_test, masks_test = self._load_data_batch(test_manifest)
        
        # Verificar que tenemos datos suficientes
        if len(X_train) == 0:
            raise ValueError("No se pudieron cargar datos de entrenamiento. Revise las rutas en manifest_train.csv")
        if len(X_val) == 0:
            raise ValueError("No se pudieron cargar datos de validación. Revise las rutas en manifest_val.csv")
        if len(X_test) == 0:
            raise ValueError("No se pudieron cargar datos de test. Revise las rutas en manifest_test.csv")
        
        print(f"Datos cargados - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test)

    def load_window_data(self, row):
        """Carga datos de una ventana específica."""
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
        
        # Extraer características (dx, dy, dz)
        input_features = input_data[['dx', 'dy', 'dz']].values
        label_features = label_data[['dx', 'dy', 'dz']].values
        mask_values = mask_data['mask'].values
        
        return input_features, label_features, mask_values
    
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

def make_masked_loss(lambda_traj: float, lambda_bias: float, eps: float = 1e-7):
    """
    Pérdida = MAE_local + lambda_traj * MSE_offset_final + lambda_bias * MSE_zero_mean
    
    Args:
        lambda_traj: Peso del término de offset final
        lambda_bias: Peso del término cero-media
        eps: Epsilon para evitar divisiones por cero
        
    Returns:
        Función de pérdida con tres términos
    """
    # Crear el contador de steps FUERA de la función de loss para evitar problemas con tf.function
    step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name='loss_step_counter')
    
    def loss_fn(y_true, y_pred):
        """
        - MAE_local: MAE en deltas (dx,dy,dz), enmascarado por timestep.
        - MSE_offset_final: MSE entre acumulados (cumsum) pred vs true en el ÚLTIMO timestep válido.
        - MSE_zero_mean: MSE del promedio por ventana del error de deltas (sesgo ≈ 0).
        """
        # y_*: [B,T,3]
        # Construir máscara robusta: 1 si timestep válido, 0 si padding.
        # Heurístico robusto: considerar padding si |y_true|==0 y añadir eps para no excluir deltas 0 reales.
        
        # --- Heurístico robusto (por defecto) ---
        abs_sum = tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True)  # [B,T,1]
        mask = tf.where(abs_sum > eps, 1.0, 0.0)                         # [B,T,1]
        # ----------------------------------------

        # # --- Máscara externa (si y_true incluye canal de máscara) ---
        # y_true_d = y_true[..., :3]
        # mask = y_true[..., 3:4]
        # y_true = y_true_d
        # # ------------------------------------------------------------

        # Aplicar máscara a deltas
        y_true_m = y_true * mask
        y_pred_m = y_pred * mask

        # 1) MAE local enmascarado (promedio por timesteps válidos)
        denom = tf.reduce_sum(mask) + eps
        mae = tf.reduce_sum(tf.abs(y_pred_m - y_true_m)) / denom

        # 2) OFFSET FINAL: cumsum por eje temporal y MSE solo en el último timestep válido
        true_accum = tf.cumsum(y_true_m, axis=1)  # [B,T,3]
        pred_accum = tf.cumsum(y_pred_m, axis=1)

        valid_counts = tf.cast(tf.reduce_sum(mask, axis=1, keepdims=True), tf.int32)  # [B,1,1]
        last_idx = tf.maximum(valid_counts - 1, 0)  # [B,1,1]

        # indices para gather_nd: [B, 2] -> (b, t_last)
        B = tf.shape(y_true)[0]
        b_inds = tf.range(B)[:, None]
        t_inds = tf.squeeze(last_idx, axis=[1,2])[:, None]
        idx = tf.concat([b_inds, t_inds], axis=1)  # [B,2]

        true_final = tf.gather_nd(true_accum, idx)  # [B,3]
        pred_final = tf.gather_nd(pred_accum, idx)  # [B,3]

        mse_final = tf.reduce_mean(tf.reduce_sum(tf.square(pred_final - true_final), axis=-1))

        # 3) CERO-MEDIA: error medio de deltas por ventana (sesgo)
        # mean_error = mean_t( y_pred - y_true ) en timesteps válidos
        # Usamos promedio ponderado por máscara para evitar padding
        # sum_t(err * mask) / sum_t(mask)
        err = (y_pred - y_true) * mask
        sum_err = tf.reduce_sum(err, axis=1)             # [B,3]
        sum_m = tf.reduce_sum(mask, axis=1) + eps        # [B,1]
        mean_err = sum_err / sum_m                       # [B,3]
        mse_bias = tf.reduce_mean(tf.reduce_sum(tf.square(mean_err), axis=-1))

        # Pérdida total
        total_loss = mae + lambda_traj * mse_final + lambda_bias * mse_bias
        
        # ===== MONITOREO SIMPLIFICADO =====
        # Incrementar contador
        step_counter.assign_add(1)
        
        # Calcular términos ponderados para logging ocasional
        weighted_traj = lambda_traj * mse_final
        weighted_bias = lambda_bias * mse_bias
        
        # Print directo sin condicionales para evitar problemas con tf.function
        # TensorFlow optimizará automáticamente la frecuencia durante el entrenamiento
        tf.print("Step:", step_counter, 
                  "Local:", tf.round(mae * 1000000) / 1000000,
                  "TrajFinal:", tf.round(mse_final * 1000000) / 1000000,
                  "Bias:", tf.round(mse_bias * 1000000) / 1000000,
                  "Lambda*TrajFinal:", tf.round(weighted_traj * 1000000) / 1000000,
                  "Lambda*Bias:", tf.round(weighted_bias * 1000000) / 1000000,
                  "Total:", tf.round(total_loss * 1000000) / 1000000,
                  "ValidMask:", tf.round(tf.reduce_sum(mask)),
                  summarize=-1)

        return total_loss

    return loss_fn

def create_model(sequence_length=3600, n_features=3):
    """Crea el modelo de red neuronal."""
    model = Sequential([
        # Capa de enmascaramiento para ignorar padding
        Masking(mask_value=0.0, input_shape=(sequence_length, n_features)),
        
        # LSTM con 128 unidades, devuelve secuencias completas
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
    """Evalúa el modelo y calcula métricas incluyendo deriva espacial."""
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
    
    # Inicializar diccionario de resultados
    results = {
        'mae_dx_norm': mae_dx_norm,
        'mae_dy_norm': mae_dy_norm, 
        'mae_dz_norm': mae_dz_norm,
        'mae_total_norm': mae_total_norm
    }
    
    # Si tenemos estadísticas de normalización, calcular métricas en metros reales
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
        
        # Actualizar resultados
        results.update({
            'mae_dx_meters': mae_dx_meters,
            'mae_dy_meters': mae_dy_meters,
            'mae_dz_meters': mae_dz_meters,
            'mae_total_meters': mae_total_meters
        })
        
        # ===== MÉTRICAS DE DERIVA ESPACIAL =====
        print(f"\n=== MÉTRICAS DE DERIVA ESPACIAL ===")
        
        # Aplicar máscaras para integrar solo timesteps válidos
        drift_metrics = calculate_drift_metrics(y_pred_meters, y_test_meters, masks_test)
        
        print(f"Deriva final media: {drift_metrics['drift_final_mean_m']:.4f} m")
        print(f"Deriva RMS: {drift_metrics['drift_rms_m']:.4f} m") 
        print(f"Diferencia longitud trayectoria: {drift_metrics['length_diff_m']:.4f} m")
        
        # Añadir métricas de deriva a resultados
        results.update(drift_metrics)
        
    return results

def calculate_drift_metrics(y_pred_meters, y_test_meters, masks_test):
    """
    Calcula métricas de deriva espacial integrando deltas a posiciones.
    
    Args:
        y_pred_meters: Predicciones desnormalizadas [batch, time, 3]
        y_test_meters: Ground truth desnormalizado [batch, time, 3]  
        masks_test: Máscaras binarias [batch, time]
        
    Returns:
        dict: Métricas de deriva espacial
    """
    # Convertir máscaras a booleano
    valid_mask = masks_test.astype(bool)
    
    # Aplicar máscaras poniendo a cero los timesteps de padding
    y_pred_masked = y_pred_meters.copy()
    y_test_masked = y_test_meters.copy()
    
    for i in range(y_pred_meters.shape[0]):  # Para cada secuencia en el batch
        for t in range(y_pred_meters.shape[1]):  # Para cada timestep
            if not valid_mask[i, t]:
                y_pred_masked[i, t, :] = 0
                y_test_masked[i, t, :] = 0
    
    # Integrar deltas para obtener posiciones (cumsum a lo largo del eje temporal)
    pos_pred = np.cumsum(y_pred_masked, axis=1)  # [batch, time, 3]
    pos_true = np.cumsum(y_test_masked, axis=1)  # [batch, time, 3]
    
    # ===== 1. DERIVA FINAL MEDIA =====
    # Distancia euclidiana entre posiciones finales (último timestep válido por secuencia)
    final_drifts = []
    
    for i in range(pos_pred.shape[0]):
        # Encontrar último timestep válido para esta secuencia
        valid_times = np.where(valid_mask[i])[0]
        if len(valid_times) > 0:
            last_valid_t = valid_times[-1]
            # Calcular distancia euclidiana solo en x,y (ignorar z)
            drift_xy = np.linalg.norm(pos_pred[i, last_valid_t, :2] - pos_true[i, last_valid_t, :2])
            final_drifts.append(drift_xy)
    
    drift_final_mean_m = np.mean(final_drifts) if final_drifts else 0.0
    
    # ===== 2. DERIVA RMS =====
    # RMS de distancias euclidianas a lo largo de toda la trayectoria
    all_drifts = []
    
    for i in range(pos_pred.shape[0]):
        for t in range(pos_pred.shape[1]):
            if valid_mask[i, t]:
                # Distancia euclidiana en cada timestep válido (solo x,y)
                drift_xy = np.linalg.norm(pos_pred[i, t, :2] - pos_true[i, t, :2])
                all_drifts.append(drift_xy)
    
    drift_rms_m = np.sqrt(np.mean(np.array(all_drifts)**2)) if all_drifts else 0.0
    
    # ===== 3. DIFERENCIA DE LONGITUD DE TRAYECTORIA =====
    # Diferencia entre longitudes totales de trayectorias predicha vs verdadera
    length_diffs = []
    
    for i in range(y_pred_meters.shape[0]):
        # Calcular longitud total de trayectoria predicha
        pred_lengths = []
        true_lengths = []
        
        for t in range(y_pred_meters.shape[1]):
            if valid_mask[i, t]:
                # Norma euclidiana del delta en este timestep (solo x,y)
                pred_step_length = np.linalg.norm(y_pred_masked[i, t, :2])
                true_step_length = np.linalg.norm(y_test_masked[i, t, :2])
                pred_lengths.append(pred_step_length)
                true_lengths.append(true_step_length)
        
        if pred_lengths and true_lengths:
            total_pred_length = np.sum(pred_lengths)
            total_true_length = np.sum(true_lengths)
            length_diff = abs(total_pred_length - total_true_length)
            length_diffs.append(length_diff)
    
    length_diff_m = np.mean(length_diffs) if length_diffs else 0.0
    
    return {
        'drift_final_mean_m': drift_final_mean_m,
        'drift_rms_m': drift_rms_m,
        'length_diff_m': length_diff_m
    }

def train_model(dataset, model_config, fast_mode=False):
    """
    Entrena usando dataset pre-dividido por sets (train/val/test).
    
    Args:
        dataset: GPSTrackDataset
        model_config: Diccionario con configuración de entrenamiento
        fast_mode: Si True, reduce datos y épocas para pruebas rápidas
        
    Returns:
        tuple: (model, history, metrics, training_time)
    """
    print("\n=== ENTRENAMIENTO DE RED NEURONAL ===")
    
    # Cargar datos por sets
    (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test) = dataset.load_by_sets()
    
    # Aplicar limitaciones de modo rápido
    if fast_mode:
        print("🚀 MODO RÁPIDO: Limitando datos para pruebas")
        max_samples = 100
        
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
            masks_train = masks_train[indices]
            
        if len(X_val) > max_samples // 4:
            indices = np.random.choice(len(X_val), max_samples // 4, replace=False)
            X_val = X_val[indices]
            y_val = y_val[indices]
            masks_val = masks_val[indices]
            
        if len(X_test) > max_samples // 4:
            indices = np.random.choice(len(X_test), max_samples // 4, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
            masks_test = masks_test[indices]
        
        print(f"Datos limitados - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Crear modelo
    model = create_model(sequence_length=X_train.shape[1], n_features=X_train.shape[2])
    model.compile(
        optimizer=Adam(learning_rate=model_config['learning_rate']),
        loss=make_masked_loss(model_config['lambda_traj'], model_config['lambda_bias']),
        metrics=['mae']
    )
    
    print(f"Modelo creado:")
    print(f"  - Secuencia: {X_train.shape[1]} timesteps")
    print(f"  - Features: {X_train.shape[2]}")
    print(f"  - Lambda traj: {model_config['lambda_traj']}")
    print(f"  - Lambda bias: {model_config['lambda_bias']}")
    
    # Crear directorio models si no existe
    os.makedirs('models', exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=model_config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/model_best.keras',
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
    
    # Entrenamiento
    start_time = time.time()
    
    print(f"\n🔄 Iniciando entrenamiento:")
    print(f"  - Épocas: {model_config['epochs']}")
    print(f"  - Batch size: {model_config['batch_size']}")
    print(f"  - Learning rate: {model_config['learning_rate']}")
    print(f"  - Patience: {model_config['patience']}")
    
    history = model.fit(
        X_train, y_train,
        batch_size=model_config['batch_size'],
        epochs=model_config['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Mostrar resumen del entrenamiento
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    epochs_trained = len(history.history['loss'])
    
    print(f"\n✅ Entrenamiento completado:")
    print(f"  - Épocas entrenadas: {epochs_trained}/{model_config['epochs']}")
    print(f"  - Tiempo total: {training_time/60:.2f} minutos")
    print(f"  - Loss final train: {final_train_loss:.6f}")
    print(f"  - Loss final val: {final_val_loss:.6f}")
    
    # Graficar historia
    plot_training_history(history, "training_history.png")
    
    # Evaluación en test
    print(f"\n📊 Evaluando en conjunto TEST...")
    test_metrics = evaluate_model(model, X_test, y_test, masks_test, norm_stats=dataset.norm_stats)
    
    # Guardar modelo final
    model.save('models/model_final.keras')
    print(f"Modelo final guardado en: models/model_final.keras")
    
    return model, history, test_metrics, training_time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Entrenar red neuronal de corrección GPS',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos principales
    parser.add_argument('--data_root', type=str, default='data/input',
                        help='Directorio raíz de datos')
    
    # Hiperparámetros de entrenamiento
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número máximo de épocas')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Tamaño de batch')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate inicial')
    parser.add_argument('--patience', type=int, default=15,
                        help='Paciencia para early stopping')
    parser.add_argument('--lambda_traj', type=float, default=1e-4,
                        help='Peso del término de offset final en la loss')
    parser.add_argument('--lambda_bias', type=float, default=0.1,
                        help='Peso del término cero-media en la loss')
    
    # Opciones adicionales
    parser.add_argument('--fast', action='store_true',
                        help='Modo rápido: reduce épocas y datos para pruebas')
    
    return parser.parse_args()

def main():
    """Función principal de entrenamiento."""
    print("=== ENTRENAMIENTO DE RED NEURONAL PARA CORRECCIÓN DE TRACKS GPS ===\n")
    
    try:
        # Parse argumentos
        args = parse_args()
        
        print(f"Configuración:")
        print(f"  - Directorio datos: {args.data_root}")
        print(f"  - Épocas: {args.epochs}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Learning rate: {args.lr}")
        print(f"  - Patience: {args.patience}")
        print(f"  - Lambda traj: {args.lambda_traj}")
        print(f"  - Lambda bias: {args.lambda_bias}")
        print(f"  - Modo rápido: {args.fast}")
        
        # Configuración del modelo
        model_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'patience': args.patience,
            'lambda_traj': args.lambda_traj,
            'lambda_bias': args.lambda_bias
        }
        
        # Aplicar limitaciones de modo rápido
        if args.fast:
            print(f"\n🚀 MODO RÁPIDO ACTIVADO")
            model_config['epochs'] = min(model_config['epochs'], 10)
            model_config['batch_size'] = min(model_config['batch_size'], 16)
            model_config['patience'] = min(model_config['patience'], 5)
            print(f"  - Épocas limitadas a: {model_config['epochs']}")
            print(f"  - Batch size limitado a: {model_config['batch_size']}")
            print(f"  - Patience limitada a: {model_config['patience']}")
        
        # Cargar dataset y entrenar
        print(f"\n📂 Cargando dataset...")
        dataset = GPSTrackDataset(data_dir=args.data_root)
        
        print(f"\n🎯 Ejecutando entrenamiento")
        model, history, test_metrics, training_time = train_model(
            dataset, model_config, fast_mode=args.fast
        )
        
        # Guardar resultados
        mode_suffix = "_fast" if args.fast else "_complete"
        results_file = f'training_results{mode_suffix}.json'
        
        results = {
            'config': model_config,
            'test_metrics': test_metrics,
            'training_time_minutes': training_time / 60,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== RESULTADOS FINALES ===")
        print(f"Modo: {'RÁPIDO' if args.fast else 'COMPLETO'}")
        if 'mae_total_meters' in test_metrics:
            print(f"MAE total (metros): {test_metrics['mae_total_meters']:.4f} m")
            if 'drift_final_mean_m' in test_metrics:
                print(f"Deriva final media: {test_metrics['drift_final_mean_m']:.4f} m")
                print(f"Deriva RMS: {test_metrics['drift_rms_m']:.4f} m")
        else:
            print(f"MAE total (normalizado): {test_metrics['mae_total_norm']:.6f}")
        print(f"Tiempo total: {training_time/60:.2f} minutos")
        print(f"Resultados guardados en: {results_file}")
        
        print(f"\n✅ Entrenamiento completado exitosamente")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())