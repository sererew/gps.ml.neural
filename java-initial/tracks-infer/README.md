# tracks-infer

CLI tool for GPS track inference using trained neural network models.

## Description

This tool performs inference on GPS tracks using trained neural network models. It supports both single model predictions and ensemble predictions with uncertainty estimation.

## Usage

### Single Model Inference
```bash
java -jar tracks-infer-1.0.0-SNAPSHOT.jar \
  --model ./model/model.zip \
  --scaler ./model/mu_sigma.json \
  --gpx ./track.gpx \
  --step 1.0
```

### Ensemble Inference (3-5 models)
```bash
java -jar tracks-infer-1.0.0-SNAPSHOT.jar \
  --model ./model1/model.zip \
  --model ./model2/model.zip \
  --model ./model3/model.zip \
  --scaler ./model/mu_sigma.json \
  --gpx ./track.gpx \
  --step 1.0
```

## Parameters

- `--model <path>`: Path to trained model file (can be repeated for ensemble)
- `--scaler <path>`: Path to mu_sigma.json normalization parameters file (required)
- `--gpx <file>`: Path to GPX file to process (required)
- `--step <meters>`: Resampling step size in meters (default: 1.0)
- `--filter <type>`: Altitude filter - `median`, `sgolay`, or `none` (default: none)
- `--maxlen <length>`: Maximum sequence length for padding (default: 5000)

## Processing Pipeline

The tool follows this processing pipeline:

1. **Load GPX**: Reads GPS points from the GPX file
2. **Convert to UTM**: Automatically detects UTM zone and converts coordinates
3. **Filter altitude** (optional): Applies median or Savitzky-Golay filtering to elevation
4. **Resample 3D**: Resamples track at uniform 3D distances
5. **Extract features**: Computes dh, dz, slope for each segment
6. **Normalize**: Applies Z-score normalization using mu_sigma.json
7. **Pad sequence**: Pads to maximum length with masking
8. **Predict**: Runs inference through neural network(s)
9. **Ensemble** (if multiple models): Averages predictions and computes uncertainty

## Output

### Text Output
```
=== INFERENCE RESULTS ===
dist_total_m: 12340.5
desnivel_pos_m: 452.1
desnivel_neg_m: 448.7
uncertainty_sigma: {
  dist: 80.2
  up: 12.0
  down: 10.5
}
```

### JSON Output
The tool also generates both console JSON output and saves results to a file:

**Single Model:**
```json
{
  "dist_total_m": 12340.5,
  "desnivel_pos_m": 452.1,
  "desnivel_neg_m": 448.7
}
```

**Ensemble (with uncertainty):**
```json
{
  "dist_total_m": 12340.5,
  "desnivel_pos_m": 452.1,
  "desnivel_neg_m": 448.7,
  "uncertainty_sigma": {
    "dist": 80.2,
    "up": 12.0,
    "down": 10.5
  }
}
```

## Output Metrics

- **dist_total_m**: Total horizontal distance traveled (meters)
- **desnivel_pos_m**: Total positive elevation gain (meters)
- **desnivel_neg_m**: Total negative elevation loss (meters)
- **uncertainty_sigma**: Standard deviation between ensemble models (only for multiple models)
  - **dist**: Distance uncertainty (meters)
  - **up**: Positive elevation uncertainty (meters)  
  - **down**: Negative elevation uncertainty (meters)

## Examples

### Basic inference with single model
```bash
java -jar tracks-infer-1.0.0-SNAPSHOT.jar \
  --model ./trained_model.zip \
  --scaler ./mu_sigma.json \
  --gpx ./my_track.gpx
```

### Ensemble with 5 models and altitude filtering
```bash
java -jar tracks-infer-1.0.0-SNAPSHOT.jar \
  --model ./model1.zip \
  --model ./model2.zip \
  --model ./model3.zip \
  --model ./model4.zip \
  --model ./model5.zip \
  --scaler ./mu_sigma.json \
  --gpx ./noisy_track.gpx \
  --filter sgolay \
  --step 1.5
```

### High-resolution processing
```bash
java -jar tracks-infer-1.0.0-SNAPSHOT.jar \
  --model ./model.zip \
  --scaler ./mu_sigma.json \
  --gpx ./long_track.gpx \
  --step 0.5 \
  --maxlen 10000
```

## Model Requirements

- Models must be saved in DL4J format (.zip files)
- All models in an ensemble should have the same architecture
- The mu_sigma.json file must correspond to the training data normalization
- Models should expect input shape: [batch, 3, sequence_length] where features are [dh, dz, slope]

## Build

To build the fat JAR:

```bash
mvn clean package
```

The executable JAR will be created at:
`target/tracks-infer-1.0.0-SNAPSHOT.jar`

## Integration

This tool is designed to work with models trained using the `tracks-train` module and data preprocessed with the `tracks-preprocess` module.