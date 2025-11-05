# Plan de Proyecto: Sistema de Filtrado de Tracks GPS Ruidosos con Red Neuronal

## 1️⃣ Objetivo
Desarrollar un sistema de **filtrado de tracks GPS** que, **dado cualquier track ruidoso**, produzca un track limpio filtrado del cual se puedan calcular con precisión:

* **Distancia total del recorrido**  
* **Desnivel positivo acumulado**  
* **Desnivel negativo acumulado**

**Enfoque**: La red neuronal actúa como **filtro punto a punto**, no predice directamente las métricas finales. Esto permite entrenar con pares (track_ruidoso, track_limpio) y después calcular cualquier métrica del track filtrado.

## 2️⃣ Datos de partida
* **17 familias de recorridos reales** (reagrupadas automáticamente):  
  * Cada familia tiene 1 **track patrón limpio** (referencia gold standard)
  * Y entre 1-16 **grabaciones ruidosas** del mismo recorrido
  * **Familias con derivadas**: Las pasadas con sufijo letra (4b, 4c, 4d, 15a, 15b, etc.) son grabaciones parciales de la misma familia (GPS sin batería) que se reagrupan automáticamente con su familia base
* **Total**: 820 ventanas de entrenamiento de diferentes grabaciones ruidosas con sus patrones de referencia

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

## 4️⃣ Arquitectura de la Red Neuronal (implementada)
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

## 5️⃣ Validación LOFO (Leave-One-Family-Out) - Implementada
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

## 6️⃣ Flujo de Uso del Sistema
**Para un track nuevo**:
1. **Preprocesar** (igual que entrenamiento): resamplear → calcular deltas → normalizar
2. **Aplicar filtro neuronal**: track_ruidoso → track_filtrado  
3. **Calcular métricas** del track filtrado:
   - Integrar deltas filtrados para obtener posiciones
   - Distancia total = suma de distancias entre puntos consecutivos
   - Desnivel+ = suma de `max(dz_filtrado, 0)`
   - Desnivel- = suma de `max(-dz_filtrado, 0)`

## 7️⃣ Implementación y Archivos
**Scripts Python** (directorio raíz):
- `1_resample_recordings.py` → `5_generate_input_dataset.py`: Preprocesamiento completo
- `6_train_neural_network.py`: Entrenamiento con LOFO y modelo final

**Datos generados**:
- `data/input/slices/`: Ventanas de entrada (grabaciones ruidosas)
- `data/input/labels/`: Ventanas de etiquetas (patrones limpios)  
- `data/input/masks/`: Máscaras binarias para padding
- `data/input/norm_stats.json`: Estadísticas de normalización
- `data/input/manifest.csv`: Metadatos de todas las ventanas

**Resultados**:
- `final_model.h5`: Modelo entrenado con todas las familias
- `complete_training_results.json`: Métricas LOFO + modelo final
- `final_model_history.png`: Gráficos de entrenamiento

## 8️⃣ Diferencias clave vs plan original
**Cambios fundamentales basados en los datos reales**:

1. **Enfoque**: 
   - ❌ **Original**: Red predice métricas globales directamente
   - ✅ **Actual**: Red filtra deltas punto a punto → calcular métricas después

2. **Familias**:
   - ❌ **Original**: "11 familias" fijas
   - ✅ **Actual**: 17 familias con reagrupación automática de derivadas

3. **Features**:
   - ❌ **Original**: `[dh, dz, pendiente]` 
   - ✅ **Actual**: `[dx, dy, dz]` (deltas 3D directos)

4. **Ventanas**:
   - ❌ **Original**: Tracks completos de longitud variable
   - ✅ **Actual**: Ventanas fijas 3600 puntos con solapamiento

**Razones de los cambios**:
- **Entrenable**: Pares (entrada_ruidosa, salida_limpia) son fáciles de obtener
- **Generalizable**: El filtro aprende patrones de ruido, no rutas específicas  
- **Flexible**: Cualquier métrica se puede calcular del track filtrado
- **Robusto**: LOFO con familias reagrupadas es estadísticamente válido

## 9️⃣ Métricas de Evaluación
**Durante entrenamiento**:
- MAE en deltas normalizados (para convergencia)
- MAE en metros reales (para interpretabilidad)

**Esperado en producción**:
- MAE de ~0.2 metros en X,Y sería excelente (mejor que precisión GPS típica)
- MAE de ~0.04 metros en Z sería muy bueno para altitud
- Error en métricas finales dependerá de la integración de estos deltas

**Interpretación del MAE**:
- MAE normalizado: Para comparar componentes (dx vs dy vs dz)
- MAE en metros: Error físico real del filtro por punto
- Desnormalización: `valor_metro = valor_norm * std + mean` usando `norm_stats.json`

### Resultado esperado
Con este sistema de filtrado:
* **Cualquier track GPS ruidoso** → **Track filtrado** → **Métricas precisas**
* Error (MAE) mejor que métodos clásicos de suavizado
* Capacidad de generalización verificada por LOFO en 17 familias reales
* Sistema robusto preparado para tracks de recorridos completamente nuevos