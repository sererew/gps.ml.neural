# Plan de Proyecto: Predicción de Métricas en Tracks de GPS Ruidosos

## 1️⃣ Objetivo
Desarrollar un sistema que, **dado cualquier track de GPS ruidoso**, prediga con la mayor precisión posible:

* **Distancia total del recorrido**  
* **Desnivel positivo acumulado**  
* **Desnivel negativo acumulado**

…sin disponer del patrón limpio en el momento de la predicción.

## 2️⃣ Datos de partida
* 11 **familias de recorridos**:  
  * cada familia tiene 1 **track patrón limpio** (con los tres valores de referencia),  
  * y 10–12 **grabaciones ruidosas** del mismo recorrido.
* Total: unas **110–115 grabaciones** ruidosas con su patrón de referencia.

## 3️⃣ Preprocesamiento (idéntico en entrenamiento y en uso real)
1. **Conversión a coordenadas métricas**  
   - Transformar lat/long a **UTM (m)** para tener X, Y y altitud en la misma unidad.
2. **Remuestreo por distancia real**  
   - Interpolar el track a **1 m de recorrido 3D** (distancia que incluye la componente vertical).
   - Así cada segmento consecutivo mide exactamente 1 m de recorrido.
3. **Cálculo de *features* por segmento**  
   Para cada metro:
   - `dh`: distancia horizontal (m),
   - `dz`: cambio de altitud (m),
   - opcional: `pendiente = dz / (dh + ε)`.
4. **Normalización (Z-score)**  
   - Calcular media y desviación de cada feature en el conjunto de *train*,
   - Transformar cada valor:  \\(x-\mu)/\sigma\\,
   - Guardar `μ` y `σ` para aplicarlos también en validación, test y en producción.
5. **Gestión de longitudes variables**  
   - Los tracks resultan en distinto nº de puntos.  
   - Se usa **padding al final + máscara** hasta una longitud máxima, para que la red ignore el relleno.

## 4️⃣ Modelo de referencia
* **Arquitectura**
  1. **Capa GRU/LSTM** de **128 unidades**: lee la secuencia punto a punto y mantiene un estado de 128 valores que resume todo el track.
  2. **Capa densa** de **64 neuronas** con activación ReLU: combina ese resumen de forma no lineal.
  3. **Capa de salida** de **3 neuronas lineales**: produce `[distancia, desnivel+, desnivel−]`.
* **Entrenamiento**
  - **Función de error**: MAE o Huber (posible ponderación para dar más peso a los desniveles).
  - **Optimizador**: Adam (learning rate ≈ 1e-3).
  - **Regularización**: Dropout 0.1–0.2 en la capa densa.
  - **Early stopping** según error de validación.

## 5️⃣ Validación de la capacidad de generalización
* **Leave-One-Family-Out (LOFO)**:
  - En cada ronda se dejan fuera **todas las grabaciones de una familia** para test,
  - Se entrena con las 10 familias restantes,
  - Se mide MAE en metros para distancia, desnivel+ y desnivel−.
  - Se repite para las 11 familias y se calcula **media y desviación** de los 11 errores.
* Esto garantiza que el modelo funciona en **recorridos nunca vistos**.
* **Modelo final**:
  - Tras comprobar el rendimiento, se entrena una última vez con **todas las familias** para tener el modelo definitivo de producción.

## 6️⃣ Baseline (método de referencia clásico)
Para saber si merece la pena usar la red:
* Suavizar la altitud con un **filtro de mediana o Savitzky–Golay**,
* Calcular:
  - Distancia total = suma de `dh`,
  - Desnivel + = suma de `max(dz, 0)`,
  - Desnivel − = suma de `max(−dz, 0)`.
* Comparar el **MAE del modelo** con el de este método:  
  el modelo debe **mejorar claramente**, sobre todo en los desniveles.

## 7️⃣ Refuerzos opcionales
* **Data augmentation** (solo en entrenamiento):  
  - Jitter en `dz`, outliers de altitud, eliminar puntos aleatoriamente antes del remuestreo, para que la red aprenda a tolerar ruido.
* **Estimación de incertidumbre**:
  - **Ensemble**: entrenar 3–5 modelos (misma arquitectura o mezclando CNN/Transformer) y promediar sus salidas.  
    La desviación estándar de sus predicciones = ±1 σ de confianza.
  - (Alternativa) **Monte Carlo Dropout**: mantener dropout activo en inferencia y repetir varias pasadas para obtener una desviación.

## 8️⃣ Hardware y entorno
* El modelo es pequeño (≈0,5–1 M de parámetros):
  - Entrenamiento e inferencia son viables en **CPU de escritorio o portátil moderno**.
  - GPU opcional si se quiere acelerar el entrenamiento.
* Implementación en **Java**:
  - **DL4J** o **DJL** soportan LSTM/GRU, padding con máscara y, si se desea, GPU.

### Resultado esperado
Con este flujo:
* Para cualquier **track de GPS nuevo y ruidoso**,  
* el sistema devuelve **distancia total, desnivel positivo y negativo** con un error (MAE) mejor que el baseline clásico,  
* y, si se usa un *ensemble* o MC Dropout, también un **rango de confianza** en esas predicciones.
