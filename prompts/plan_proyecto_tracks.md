# Plan de Proyecto: Sistema de Filtrado de Tracks GPS Ruidosos con Red Neuronal

## 1️⃣ Objetivo
Desarrollar un sistema de **filtrado de tracks GPS** que, **dado cualquier track ruidoso**, produzca un track limpio filtrado del cual se puedan calcular con precisión:

* **Distancia total del recorrido**  
* **Desnivel positivo acumulado**  
* **Desnivel negativo acumulado**

**Enfoque**: La red neuronal actúa como **filtro punto a punto**, no predice directamente las métricas finales. Esto permite entrenar con pares (track_ruidoso, track_limpio) y después calcular cualquier métrica del track filtrado.

## 2️⃣ Datos de partida
* **25 pasadas de recorridos reales** (reagrupadas automáticamente en **17 familias** para entrenamiento):  
  * Cada familia tiene 1 **track patrón limpio** (referencia gold standard)
  * Y entre 1-16 **grabaciones ruidosas** del mismo recorrido
  * **Familias con derivadas**: 8 pasadas son versiones cortas por batería agotada de GPS:
    - **Familia 4**: `['4', '4b', '4c', '4d']` - 4 pasadas que forman 1 familia
    - **Familia 15**: `['15', '15a', '15b', '15c', '15d']` - 5 pasadas que forman 1 familia
  * **Familias individuales**: Las otras 16 pasadas son familias independientes
* **Total**: 255 tracks procesados y 820 ventanas de entrenamiento de **17 familias**

## 3️⃣ Preprocesamiento (implementado en scripts 1-5)
1. **Resampleado temporal a 1 Hz** de todas las grabaciones
2. **Alineación temporal** entre grabaciones ruidosas y patrones limpios  
3. **Conversión a coordenadas métricas locales** (x,y,z en metros)
4. **Cálculo de deltas**: `dx`, `dy`, `dz` entre puntos consecutivos
5. **Ventanas deslizantes**:
   - Ventanas de **3600 puntos (1 hora)** con solape de 1800 puntos (30 min)
   - **Padding con ceros** y **máscara binaria** para longitudes variables
6. **Normalización Z-score global**:
   - Calculada sobre todas las grabaciones: `mean` y `std` por componente
   - Guardada en `norm_stats.json` para desnormalización posterior

## 4️⃣ Arquitectura de la Red Neuronal (implementada en script 6)
**Red como filtro secuencia-a-secuencia**:
* **Entrada**: `[batch, time, 3]` donde `3 = [dx, dy, dz]` normalizados del track ruidoso
* **Salida**: `[batch, time, 3]` donde `3 = [dx_filtrado, dy_filtrado, dz_filtrado]`

**Capas**:
1. **Masking(0.0)**: Ignora padding automáticamente
2. **LSTM(128)** con `return_sequences=True` + dropout 0.1: Procesa secuencia completa  
3. **Dense(64)** con ReLU + dropout 0.2: Combinación no lineal
4. **Dense(3)** lineal: Salida de deltas filtrados

**Entrenamiento**:
- **Loss**: MAE enmascarado (ignora posiciones con padding)
- **Optimizador**: Adam (lr=1e-3)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Métricas**: MAE en valores normalizados y convertidos a metros reales

## 5️⃣ Validación LOFO (Leave-One-Family-Out) - (implementada script 6)
**Reagrupación automática de familias**:
- **Familia 4**: Incluye automáticamente `['4', '4b', '4c', '4d']` 
- **Familia 15**: Incluye automáticamente `['15', '15a', '15b', '15c', '15d']`
- **Otras familias**: Permanecen individuales `['1'], ['2'], ['3']`, etc.

**Proceso LOFO**:
- **17 rondas** (una por familia reagrupada)
- En cada ronda: familia completa para test, resto (16 familias) para train/validation
- **Garantiza** que el modelo funciona en recorridos completamente nuevos
- **Métricas**: MAE promedio ± desviación estándar en deltas filtrados

**Ventajas del reagrupamiento**:
- Evita sesgo de familias con 1 sola grabación vs 16 grabaciones
- LOFO estadísticamente válido (todas las familias aportan datos suficientes)
- No hay "data leakage" entre grabaciones de la misma familia

## 6️⃣ Sistema de Filtrado Individual (scripts 7_*.py)
**Filtros implementados**:

### **7_nn_filter.py** - Filtro de Red Neuronal
- **Proceso completo**: Deltas → Normalización → Red → Desnormalización → Integración
- **Características**:
  - Carga modelo entrenado (`final_model.h5`) y estadísticas (`norm_stats.json`)
  - Maneja tracks largos en chunks de 3600 puntos
  - Conversión robusta lat/lon ↔ coordenadas métricas
  - Integración correcta de deltas para preservar posición inicial

### **Filtros de Referencia Clásicos**:
- **7_identity_filter.py**: Sin filtrado (baseline de comparación)
- **7_moving_average_filter.py**: Media aritmética en ventana deslizante
- **7_triangular_weighted_filter.py**: Media ponderada con pesos triangulares
- **7_median_filter.py**: Mediana móvil (efectivo contra outliers)
- **7_savgol_filter.py**: Savitzky-Golay con ajuste polinomial local
- **7_exponential_filter.py**: Suavizado exponencial (EMA)
- **7_gaussian_filter.py**: Filtro gaussiano con kernel configurable
- **7_kalman_filter.py**: Filtro Kalman simple con modelo de velocidad constante

**Características comunes**:
- **Uso consistente**: `python 7_<filtro>_filter.py input.gpx [output.gpx]`
- **Generación automática** de nombres: `<original>_<filtro>_filtered.gpx`
- **Creación automática** de directorios de salida
- **Parámetros configurables** específicos por filtro
- **Compatibilidad Windows**: Sin caracteres Unicode problemáticos

## 7️⃣ Procesamiento Masivo (script 8_apply_all_filters.py)
**Funcionalidad**:
- **Detección automática** de todos los tracks en `data/preprocessed/<pasada>/`
- **Detección automática** de todos los filtros disponibles (scripts `7_*_filter.py`)
- **Aplicación masiva** de todos los filtros a todos los tracks
- **Procesamiento paralelo** configurable (4 procesos por defecto)

**Características avanzadas**:
- **Gestión inteligente**: No sobrescribe archivos existentes (configurable)
- **Estructura organizada**: `data/filtered/<filtro>/<pasada>/`
- **Progress tracking**: Progreso en tiempo real con estadísticas
- **Manejo robusto** de errores y timeouts (5 min por filtro)
- **Escalabilidad**: Procesa automáticamente 25 pasadas × 9 filtros = 2295 combinaciones

**Uso**:
```bash
# Procesar todo
python 8_apply_all_filters.py

# Solo ciertas pasadas/filtros
python 8_apply_all_filters.py --pasadas 1,2,3 --filtros nn,kalman,savgol

# Con sobrescritura
python 8_apply_all_filters.py --overwrite
```

## 8️⃣ Análisis Comparativo (script 9_compare_tracks.py)
**Funcionalidad principal**:
- **Comparación automática** de todos los tracks filtrados vs sus patrones de referencia
- **Recorte temporal**: Solo compara puntos dentro del rango temporal del patrón
- **Métricas completas**: Distancia, desniveles, desviación 3D punto a punto
- **Salida Excel** con análisis detallado y resumen por filtro

**Métricas calculadas**:

### **Métricas del patrón (referencia)**:
- `total_pattern_length`: Distancia total del patrón
- `total_pattern_elevation_gain/loss`: Desniveles del patrón
- `total_pattern_elevation_gain/loss_threshold`: Con umbral de 5m

### **Desviaciones respecto al patrón**:
- `total_length_deviation`: Diferencia en distancia total
- `total_elevation_gain/loss_deviation`: Diferencias en desniveles
- `total_elevation_gain/loss_deviation_threshold`: Con umbral de 5m
- `mean_point_deviation`: Desviación 3D media punto a punto (metros)
- `std_point_deviation`: Desviación estándar de la desviación 3D

**Recorte temporal crítico**:
- **Problema**: Algunos tracks empiezan antes/terminan después que el patrón
- **Solución**: `trim_track_to_pattern_timerange()` limita comparación al rango del patrón
- **Garantía**: Todas las métricas se calculan solo en el tiempo válido

**Salida Excel**:
- **Track_Comparison**: Resultados detallados track por track
- **Filter_Summary**: Estadísticas agregadas por filtro
- **Formato profesional**: Headers, formato numérico, columnas ajustadas

## 9️⃣ Flujo Completo del Sistema
**Pipeline completo de evaluación**:

1. **Scripts 1-5**: Preprocesamiento y generación de dataset
2. **Script 6**: Entrenamiento LOFO y modelo final
3. **Scripts 7**: Implementación de 9 filtros (neuronal + clásicos)
4. **Script 8**: Aplicación masiva de filtros (2295 combinaciones)
5. **Script 9**: Análisis comparativo y reporte Excel

**Para un track nuevo**:
1. **Preprocesar**: Resamplear a 1Hz → calcular deltas → normalizar
2. **Aplicar filtro**: Cualquiera de los 9 filtros implementados
3. **Calcular métricas**: Del track filtrado (distancia, desniveles, etc.)

## 🔟 Implementación y Archivos
**Scripts Python** (directorio raíz):
- **Preprocesamiento**: `1_resample_recordings.py` → `5_generate_input_dataset.py`
- **Entrenamiento**: `6_train_neural_network.py`
- **Filtros individuales**: `7_identity_filter.py` → `7_nn_filter.py` (9 filtros)
- **Procesamiento masivo**: `8_apply_all_filters.py`
- **Análisis comparativo**: `9_compare_tracks.py`

**Estructura de datos**:
```
data/
├── input/                    # Dataset de entrenamiento
│   ├── slices/              # Ventanas de entrada (grabaciones ruidosas)
│   ├── labels/              # Ventanas de etiquetas (patrones limpios)
│   ├── masks/               # Máscaras binarias para padding
│   ├── norm_stats.json      # Estadísticas de normalización
│   └── manifest.csv         # Metadatos de ventanas
├── preprocessed/            # Tracks resampleados a 1Hz
│   └── <pasada>/
│       ├── <pasada>_aligned_pattern_resampled.gpx  # Patrón de referencia
│       └── *_resampled.gpx  # Grabaciones resampleadas
└── filtered/                # Tracks filtrados por método
    └── <filtro>/
        └── <pasada>/
            └── *_<filtro>_filtered.gpx
```

**Resultados del entrenamiento**:
- `final_model.h5`: Modelo neuronal entrenado
- `complete_training_results.json`: Métricas LOFO completas
- `final_model_history.png`: Gráficos de convergencia

**Resultados del análisis**:
- `track_comparison_results.xlsx`: Comparación completa de todos los filtros

## 1️⃣1️⃣ Métricas de Evaluación
**Durante entrenamiento**:
- MAE en deltas normalizados (convergencia)
- MAE en metros reales (interpretabilidad física)

**En producción**:
- **Desviación 3D punto a punto**: Métrica principal de precisión
- **Error en distancia total**: Acumulación de errores de trayectoria
- **Error en desniveles**: Precisión en cálculo de elevación
- **Comparación directa**: 9 filtros en condiciones idénticas

**Interpretación**:
- MAE < 0.5m en desviación 3D → Excelente para GPS típico
- Desviación relativa en distancia < 1% → Muy buena precisión
- Comparación estadística entre filtros → Identificar mejor método

## 1️⃣2️⃣ Resultado Final
Con este sistema completo:
* **✅ Entrenamiento robusto**: LOFO en 17 familias garantiza generalización
* **✅ Comparación exhaustiva**: 9 filtros evaluados en condiciones idénticas  
* **✅ Métricas completas**: Distancia, desniveles y desviación 3D
* **✅ Automatización total**: Desde track ruidoso hasta reporte Excel
* **✅ Reproducibilidad**: Pipeline completamente automatizado y documentado

**Capacidades del sistema**:
- **Cualquier track GPS ruidoso** → **Track filtrado de alta precisión**
- **Evaluación objetiva** de filtro neuronal vs métodos clásicos
- **Métricas precisas** para cálculo de distancia y desniveles
- **Sistema escalable** para nuevas pasadas y filtros adicionales