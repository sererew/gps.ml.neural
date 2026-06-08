# tracks-ml

Machine Learning project for GPS tracks processing.

## Modules

- **tracks-core**: Core library with common functionality
- **tracks-preprocess**: CLI tool for GPS tracks preprocessing
- **tracks-train**: Training module with LOFO and baseline functionality
- **tracks-infer**: CLI tool for GPS tracks inference
- **tracks-eval**: Evaluation module for reports and plots

## Requirements

- Java 21
- Maven 3.6+

## Build

To compile the entire project:

```bash
mvn -q -DskipTests package
```

This will create:
- JAR files for all modules in their respective `target/` directories
- Fat JARs for CLI modules (preprocess, train, infer) with all dependencies included

## Usage

After building, you can run the CLI tools:

```bash
# Preprocessing
java -jar tracks-preprocess/target/tracks-preprocess-1.0.0-SNAPSHOT.jar [options]

# Training
java -jar tracks-train/target/tracks-train-1.0.0-SNAPSHOT.jar [options]

# Inference
java -jar tracks-infer/target/tracks-infer-1.0.0-SNAPSHOT.jar [options]

# Evaluation
java -jar tracks-eval/target/tracks-eval-1.0.0-SNAPSHOT.jar [options]
```

## Project Structure

Base package: `uo.ml.neural.tracks.*`

```
tracks-ml/
├── tracks-core/           # Common library
├── tracks-preprocess/     # Preprocessing CLI
├── tracks-train/          # Training CLI
├── tracks-infer/          # Inference CLI
└── tracks-eval/           # Evaluation CLI
```