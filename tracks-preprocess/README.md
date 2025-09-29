# tracks-preprocess

CLI tool for preprocessing GPS tracks and generating ML training datasets.

## Description

This tool processes folders of GPX files organized by families and generates structured datasets ready for machine learning training. Each family should contain noisy GPS tracks and one pattern track (reference).

## Usage

```bash
preprocess.bat --input ./data/raw --output ./data/processed --step 1.0 --filter sgolay
```

## Parameters

- `--input <dir>`: Root directory containing family subdirectories with GPX files (required)
- `--output <dir>`: Output directory for processed datasets (required)
- `--step <meters>`: Resampling step size in meters (default: 1.0)
- `--filter <type>`: Altitude filter type - `median`, `sgolay`, or `none` (default: none)

## Input Structure

The input directory should follow this convention:

```
data/raw/
├── family1/
│   ├── track001.gpx
│   ├── track002.gpx
│   ├── track003.gpx
│   └── family1_pattern.gpx    # Reference pattern (required)
├── family2/
│   ├── noisy_track_a.gpx
│   ├── noisy_track_b.gpx
│   └── family2_pattern.gpx    # Reference pattern (required)
└── family3/
    ├── gps_001.gpx
    └── family3_pattern.gpx    # Reference pattern (required)
```

**Important**: Each family directory must contain exactly one file with `_pattern.gpx` in the name.

## Output Structure

The tool generates the following output structure:

```
data/processed/
├── features/
│   ├── family1/
│   │   ├── track001.csv       # dh,dz,slope for each step
│   │   ├── track002.csv
│   │   ├── track003.csv
│   │   └── family1_pattern.csv
│   └── family2/
│       ├── noisy_track_a.csv
│       ├── noisy_track_b.csv
│       └── family2_pattern.csv
├── labels/
│   ├── family1.csv            # dist_total,desn_pos,desn_neg
│   └── family2.csv
├── lengths/
│   ├── family1/
│   │   ├── track001.txt       # Number of steps
│   │   ├── track002.txt
│   │   ├── track003.txt
│   │   └── family1_pattern.txt
│   └── family2/
│       ├── noisy_track_a.txt
│       ├── noisy_track_b.txt
│       └── family2_pattern.txt
└── mu_sigma.json              # Global Z-score normalization parameters
```

## Processing Pipeline

For each family, the tool:

1. **Reads GPX files**: Extracts GPS points from all tracks
2. **Identifies pattern**: Finds the `*_pattern.gpx` file as reference
3. **Converts coordinates**: Transforms GPS (lat/lon) to UTM coordinates
4. **Applies altitude filtering**: Optional filtering (median or Savitzky-Golay) on elevation
5. **Resamples tracks**: 3D resampling at uniform step intervals
6. **Extracts features**: Computes geometric features (dh, dz, slope) for each segment
7. **Saves outputs**:
   - Features as CSV files (one per track)
   - Labels from pattern track (distance totals and elevation changes)
   - Sequence lengths for variable-length handling
8. **Computes normalization**: Global Z-score parameters across all noisy tracks

## Features

- **dh**: 2D horizontal distance between consecutive points (meters)
- **dz**: Vertical elevation change between consecutive points (meters)  
- **slope**: Calculated as `dz / (dh + 1e-6)` to avoid division by zero

## Labels

For each family, labels are computed from the pattern track:
- **dist_total**: Total horizontal distance traveled
- **desn_pos**: Total positive elevation gain
- **desn_neg**: Total negative elevation loss (absolute value)

## Examples

### Basic preprocessing
```bash
preprocess.bat --input ./gps_data --output ./processed_data
```

### With 2-meter resampling and median altitude filtering
```bash
preprocess.bat --input ./raw_tracks --output ./clean_tracks \
  --step 2.0 --filter median
```

### With Savitzky-Golay smoothing
```bash
preprocess.bat --input ./input_dir --output ./output_dir \
  --step 1.5 --filter sgolay
```

## Build

To build the fat JAR:

```bash
mvn clean package
```

The executable JAR will be created at:
`target/tracks-preprocess-1.0.0-SNAPSHOT.jar`