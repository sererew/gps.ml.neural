# tracks-train

CLI tools for training neural network models on GPS track datasets.

## Description

This module provides two training modes for GPS track analysis:
- **LOFO Cross-Validation**: Leave-One-Family-Out validation for model evaluation
- **Final Model Training**: Training on all available data to produce a deployable model

Both tools use LSTM neural networks to predict track metrics (distance, elevation changes) from GPS track features.

## Usage

### LOFO Cross-Validation Training
```bash
lofo-train.bat --data ./data/preprocessed --output ./lofo_results
```

### Final Model Training
```bash
final-train.bat --data ./data/preprocessed --out ./trained_model \
  --epochs 150 --lr 0.001
```

## Commands

### lofo-trainer
Performs Leave-One-Family-Out cross-validation for model evaluation.

**Parameters:**
- `--input <dir>`: Preprocessed data directory (output from tracks-preprocess) (required)
- `--output <dir>`: Output directory for LOFO results (default: ./lofo_results)
- `--epochs <num>`: Number of training epochs per fold (default: 100)
- `--lr <rate>`: Learning rate (default: 0.001)

### final-trainer
Trains a final model on all available families for deployment.

**Parameters:**
- `--input <dir>`: Preprocessed data directory (output from tracks-preprocess) (required)
- `--output <dir>`: Output directory for trained model (required)
- `--epochs <num>`: Number of training epochs (default: 150)
- `--lr <rate>`: Learning rate (default: 0.001)

## Input Structure

The input directory should be the output from tracks-preprocess:

```
data/preprocessed/
├── features/
│   ├── family1/
│   │   ├── track001.csv       # dh,dz,slope features
│   │   ├── track002.csv
│   │   └── family1_pattern.csv
│   └── family2/
│       ├── noisy_track_a.csv
│       └── family2_pattern.csv
├── labels/
│   ├── family1.csv            # dist_total,desn_pos,desn_neg
│   └── family2.csv
├── lengths/
│   ├── family1/
│   │   ├── track001.txt       # Sequence lengths
│   │   └── track002.txt
│   └── family2/
│       └── noisy_track_a.txt
└── mu_sigma.json              # Normalization parameters
```

## Output Structure

### LOFO Cross-Validation Output
```
lofo_results/
├── fold_results.csv           # Per-fold performance metrics
├── summary_report.md          # Aggregated results and statistics
├── models/                    # Trained models per fold
│   ├── fold_family1_model.zip
│   ├── fold_family2_model.zip
│   └── ...
└── predictions/               # Fold predictions
    ├── fold_family1_predictions.json
    ├── fold_family2_predictions.json
    └── ...
```

### Final Model Output
```
trained_model/
├── model.zip                  # Trained DL4J MultiLayerNetwork
├── mu_sigma.json             # Normalization parameters (copied)
└── README.md                 # Model documentation and usage
```

## Neural Network Architecture

Both training modes use the same LSTM-based architecture:

1. **LSTM Layer**: 128 units, tanh activation
2. **Global Pooling**: LAST (uses masking for variable-length sequences)
3. **Dense Layer**: 64 units, ReLU activation  
4. **Output Layer**: 3 units, linear activation

**Input Features:**
- `dh`: 2D horizontal distance (normalized)
- `dz`: Vertical elevation change (normalized)
- `slope`: Calculated slope = dz/(dh+1e-6) (normalized)

**Output Predictions:**
- Total horizontal distance traveled (meters)
- Total positive elevation gain (meters)
- Total negative elevation loss (meters, absolute)

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Mean Absolute Error (MAE)
- **Batch Processing**: All samples processed in single batch
- **Sequence Handling**: Variable-length with masking support
- **Normalization**: Z-score using preprocessed mu_sigma.json

## LOFO Cross-Validation

The LOFO trainer implements Leave-One-Family-Out cross-validation:

1. For each family:
   - Train model on all other families
   - Test on the held-out family
   - Record predictions and performance metrics
2. Aggregate results across all folds
3. Generate statistical summary with mean ± standard deviation

This provides robust model evaluation and helps detect overfitting to specific GPS track patterns.

## Examples

### Basic LOFO validation
```bash
lofo-train.bat --data ./data/preprocessed
```

### LOFO with custom parameters
```bash
lofo-train.bat --data ./processed_data --output ./validation_results \
  --epochs 200 --lr 0.0005
```

### Train final model for deployment
```bash
final-train.bat --data ./data/preprocessed --out ./production_model \
  --epochs 300 --lr 0.001
```

### Train fast model for testing
```bash
final-train.bat --data ./data/preprocessed --out ./test_model \
  --epochs 50 --lr 0.01
```

## Model Deployment

After training with `final-trainer`, the model can be used for inference:

```bash
# Using tracks-infer module
java -jar tracks-infer-1.0.0-SNAPSHOT.jar \
  --model ./trained_model/model.zip \
  --scaler ./trained_model/mu_sigma.json \
  --gpx ./new_track.gpx
```

## Build

To build the fat JAR:

```bash
mvn clean package
```

The executable JAR will be created at:
`target/tracks-train-1.0.0-SNAPSHOT.jar`

## Performance Tips

- **Memory**: Training requires sufficient heap space for large datasets (`-Xmx4g` recommended)
- **CPU**: Training is CPU-intensive; more cores improve performance
- **Epochs**: Start with default values; increase if loss hasn't converged
- **Learning Rate**: Lower values (0.0001-0.001) generally work better for GPS data

## Dependencies

- **DL4J**: Deep learning framework
- **ND4J**: N-dimensional arrays for Java
- **PicoCLI**: Command-line interface
- **Jackson**: JSON processing for results serialization