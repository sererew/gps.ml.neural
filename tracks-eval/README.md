# tracks-eval

CLI tool for evaluating LOFO validation and baseline results, generating statistical reports.

## Description

This tool analyzes the output from LOFO (Leave-One-Family-Out) cross-validation and baseline comparisons, generating comprehensive statistical reports in CSV and Markdown formats.

## Usage

```bash
java -jar tracks-eval-1.0.0-SNAPSHOT.jar \
  --results ./lofo_output --output ./evaluation_report --format both
```

## Parameters

- `--results <path>`: Directory containing LOFO results or result files (required)
- `--output <dir>`: Output directory for reports (default: ./evaluation_report)
- `--format <type>`: Output format - `csv`, `markdown`, or `both` (default: both)

## Input Sources

The tool can analyze results from various sources:

### LOFO Output Files
- Text files containing LOFO validation output
- Files with "lofo" in the name
- Console output saved to `.txt` or `.log` files

### JSON Results
- Structured JSON files with LOFO results
- Baseline comparison data in JSON format

### Baseline Files
- Files containing baseline comparison results
- Files with "baseline" in the name

## Output Reports

### CSV Report (`evaluation_report.csv`)
Structured data with:
- Individual LOFO fold results
- Aggregated statistics (mean ± std)
- Model comparison metrics
- Improvement percentages

### Markdown Report (`evaluation_report.md`)
Formatted tables including:
- Individual LOFO results table
- Neural Network vs Baseline comparison
- Aggregated statistics with mean ± standard deviation
- Summary analysis with key findings

### Summary Report (`summary.txt`)
Concise text summary with:
- Overall performance metrics
- Best and worst performing families
- Key insights and improvements

## Example Output

### Markdown Table
```markdown
| Family | Neural Network | Baseline | Improvement | Improvement % |
|--------|----------------|----------|-------------|---------------|
| family1| 245.3         | 312.7    | 67.4        | 21.6%         |
| family2| 189.4         | 287.1    | 97.7        | 34.0%         |
```

### Aggregated Statistics
```markdown
| Metric | Mean | Standard Deviation |
|--------|------|-------------------|
| Overall | 217.4 | 45.2 |
| Distance | 156.8 | 32.1 |
| Elevation (+) | 89.2 | 18.7 |
| Elevation (-) | 91.3 | 19.4 |
```

## Examples

### Analyze LOFO results directory
```bash
java -jar tracks-eval-1.0.0-SNAPSHOT.jar \
  --results ./lofo_validation_output \
  --output ./reports \
  --format both
```

### Analyze single result file
```bash
java -jar tracks-eval-1.0.0-SNAPSHOT.jar \
  --results ./lofo_results.txt \
  --output ./analysis
```

### Generate only CSV report
```bash
java -jar tracks-eval-1.0.0-SNAPSHOT.jar \
  --results ./results \
  --format csv
```

## Integration

This tool is designed to work with:
- Output from `tracks-train` LOFO validation (`LofoTrainerCli`)
- Baseline comparison results
- Custom JSON result files

## Build

The module builds as a fat JAR with all dependencies included:

```bash
mvn clean package
```

Executable JAR: `target/tracks-eval-1.0.0-SNAPSHOT.jar`