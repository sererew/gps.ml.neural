package uo.ml.neural.tracks.train.commands.lofo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.core.exception.IO;
import uo.ml.neural.tracks.train.model.FoldResult;

/**
 * Handles saving LOFO cross-validation results to the output directory structure.
 */
public class LofoResultsSaver {

    private final Path outputDir;
    private final ObjectMapper objectMapper;

    public LofoResultsSaver(Path outputDir) {
        this.outputDir = outputDir;
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Saves all LOFO results following the structure specified in README.
     */
    public void saveResults(List<FoldResult> foldResults) {
        createOutputDirectories();
        saveFoldResultsCsv(foldResults);
        saveSummaryReport(foldResults);
        saveModels(foldResults);
        savePredictions(foldResults);
        
        System.out.println("Results saved to: " + outputDir);
    }

    private void createOutputDirectories() {
        IO.exec(() -> {
            Files.createDirectories(outputDir);
            Files.createDirectories(outputDir.resolve("models"));
            Files.createDirectories(outputDir.resolve("predictions"));
        });
    }

    private void saveFoldResultsCsv(List<FoldResult> foldResults) {
        Path csvFile = outputDir.resolve("fold_results.csv");
        
        StringBuilder csv = new StringBuilder();
        csv.append("fold,test_family,train_families,")
           .append("nn_mae_distance,nn_mae_elevation_pos,nn_mae_elevation_neg,nn_mae_overall,")
           .append("baseline_mae_distance,baseline_mae_elevation_pos,baseline_mae_elevation_neg,baseline_mae_overall\n");

        for (int i = 0; i < foldResults.size(); i++) {
            FoldResult result = foldResults.get(i);
            double[] nnMAE = result.getNnMAE();
            double[] baselineMAE = result.getBaselineMAE();
            
            csv.append(String.format(Locale.US,
            		"%d,%s,\"%s\",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    i + 1,
                    result.getTestFamily(),
                    String.join(";", result.getTrainFamilies()),
                    nnMAE[0], 
                    nnMAE[1], 
                    nnMAE[2], 
                    result.getNnOverallMAE(),
                    baselineMAE[0], 
                    baselineMAE[1], 
                    baselineMAE[2], 
                    result.getBaselineOverallMAE())
            	);
        }

        IO.exec(() -> Files.writeString(csvFile, csv.toString()));
    }

    private void saveSummaryReport(List<FoldResult> foldResults) {
        Path reportFile = outputDir.resolve("summary_report.md");
        
        StringBuilder report = new StringBuilder();
        report.append("# LOFO Cross-Validation Summary Report\n\n");
        report.append("## Overview\n\n");
        report.append(String.format("- **Total Folds**: %d\n", foldResults.size()));
        report.append(String.format("- **Validation Method**: Leave-One-Family-Out\n"));
        report.append("\n## Performance Summary\n\n");
        
        // Calculate statistics
        double[] nnMeans = calculateMeans(foldResults, true);
        double[] nnStds = calculateStds(foldResults, nnMeans, true);
        double[] baselineMeans = calculateMeans(foldResults, false);
        double[] baselineStds = calculateStds(foldResults, baselineMeans, false);
        
        report.append("### Neural Network Performance\n\n");
        report.append("| Metric            | Mean ± Std | Range |\n");
        report.append("|-------------------|------------|-------|\n");
        report.append(String.format(
        		"| Distance MAE      | %.3f ± %.3f | [%.3f, %.3f] |\n",
                nnMeans[0], 
                nnStds[0], 
                getMin(foldResults, 0, true), 
                getMax(foldResults, 0, true)
            ));
        report.append(String.format(
        		"| Elevation Pos MAE | %.3f ± %.3f | [%.3f, %.3f] |\n",
                nnMeans[1], 
                nnStds[1], 
                getMin(foldResults, 1, true), 
                getMax(foldResults, 1, true)
            ));
        report.append(String.format(
        		"| Elevation Neg MAE | %.3f ± %.3f | [%.3f, %.3f] |\n",
                nnMeans[2], 
                nnStds[2], 
                getMin(foldResults, 2, true), 
                getMax(foldResults, 2, true)
            ));
        report.append(String.format(
        		"| Overall MAE       | %.3f ± %.3f | [%.3f, %.3f] |\n",
                nnMeans[3], 
                nnStds[3], 
                getMinOverall(foldResults, true), 
                getMaxOverall(foldResults, true)
            ));
        
        report.append("\n### Baseline Performance\n\n");
        report.append("| Metric            | Mean ± Std | Range |\n");
        report.append("|-------------------|------------|-------|\n");
        report.append(String.format(
        		"| Distance MAE      | %.3f ± %.3f | [%.3f, %.3f] |\n",
                baselineMeans[0], 
                baselineStds[0], 
                getMin(foldResults, 0, false), 
                getMax(foldResults, 0, false)
            ));
        report.append(String.format(
        		"| Elevation Pos MAE | %.3f ± %.3f | [%.3f, %.3f] |\n",
                baselineMeans[1], 
                baselineStds[1], 
                getMin(foldResults, 1, false), 
                getMax(foldResults, 1, false)
            ));
        report.append(String.format(
        		"| Elevation Neg MAE | %.3f ± %.3f | [%.3f, %.3f] |\n",
                baselineMeans[2], 
                baselineStds[2], 
                getMin(foldResults, 2, false), 
                getMax(foldResults, 2, false)
            ));
        report.append(String.format(
        		"| Overall MAE       | %.3f ± %.3f | [%.3f, %.3f] |\n",
                baselineMeans[3], 
                baselineStds[3], 
                getMinOverall(foldResults, false), 
                getMaxOverall(foldResults, false)
            ));
        
        report.append("\n## Fold Details\n\n");
        for (int i = 0; i < foldResults.size(); i++) {
            FoldResult result = foldResults.get(i);
            report.append(String.format(
            		"### Fold %d: %s\n\n", 
            		i + 1, 
            		result.getTestFamily()
            	));
            report.append(String.format(
            		"- **Training Families**: %s\n", 
            		String.join(", ", result.getTrainFamilies())
            	));
            report.append(String.format(
            		"- **NN MAE**: [%.3f, %.3f, %.3f] (overall: %.3f)\n",
                    result.getNnMAE()[0], 
                    result.getNnMAE()[1], 
                    result.getNnMAE()[2], 
                    result.getNnOverallMAE()
                ));
            report.append(String.format(
            		"- **Baseline MAE**: [%.3f, %.3f, %.3f] (overall: %.3f)\n\n",
                    result.getBaselineMAE()[0], 
                    result.getBaselineMAE()[1], 
                    result.getBaselineMAE()[2], 
                    result.getBaselineOverallMAE()
                ));
        }

        IO.exec(() -> Files.writeString(reportFile, report.toString()));
    }

    private void saveModels(List<FoldResult> foldResults) {
        Path modelsDir = outputDir.resolve("models");
        
        for (int i = 0; i < foldResults.size(); i++) {
            FoldResult result = foldResults.get(i);
            String fileName = String.format("fold_%s_model.zip", result.getTestFamily());
            Path modelFile = modelsDir.resolve(fileName);
            
            try {
                result.getTrainedModel().save(modelFile.toFile());
            } catch (IOException e) {
                throw new CommandException("Failed to save model for fold " 
                		+ result.getTestFamily(), e);
            }
        }
    }

    private void savePredictions(List<FoldResult> foldResults) {
        Path predictionsDir = outputDir.resolve("predictions");
        
        for (FoldResult result : foldResults) {
            String fileName = String.format(
            		"fold_%s_predictions.json", 
            		result.getTestFamily()
            	);
            Path predictionsFile = predictionsDir.resolve(fileName);
            
            PredictionData predictionData = new PredictionData(
                    result.getTestFamily(),
                    result.getTrainFamilies(),
                    result.getPredictions(),
                    result.getActualLabels()
            );
            
            IO.exec(() -> objectMapper.writeValue(
            		predictionsFile.toFile(), 
            		predictionData
            	));
        }
    }

    // Helper methods for statistics calculation
    private double[] calculateMeans(List<FoldResult> results, boolean isNeuralNetwork) {
        double[] sums = new double[4]; // 3 MAE values + overall
        
        for (FoldResult result : results) {
            double[] mae = getMae(isNeuralNetwork, result);
            double overall = getOverallMae(isNeuralNetwork, result);
            
            for (int i = 0; i < 3; i++) {
                sums[i] += mae[i];
            }
            sums[3] += overall;
        }
        
        for (int i = 0; i < 4; i++) {
            sums[i] /= results.size();
        }
        
        return sums;
    }

	private double getOverallMae(boolean isNeuralNetwork, FoldResult result) {
		return isNeuralNetwork ? result.getNnOverallMAE() : result.getBaselineOverallMAE();
	}

	private double[] getMae(boolean isNeuralNetwork, FoldResult result) {
		return isNeuralNetwork ? result.getNnMAE() : result.getBaselineMAE();
	}

    private double[] calculateStds(
    		List<FoldResult> results, 
    		double[] means, 
    		boolean isNeuralNetwork) {
    	
        double[] variances = new double[4];
        
        for (FoldResult result : results) {
            double[] mae = getMae(isNeuralNetwork, result);
            double overall = getOverallMae(isNeuralNetwork, result);
            
            for (int i = 0; i < 3; i++) {
                variances[i] += Math.pow(mae[i] - means[i], 2);
            }
            variances[3] += Math.pow(overall - means[3], 2);
        }
        
        for (int i = 0; i < 4; i++) {
            variances[i] = Math.sqrt(variances[i] / results.size());
        }
        
        return variances;
    }

    private double getMin(List<FoldResult> results, int index, boolean isNeuralNetwork) {
        return results.stream()
                .mapToDouble(r -> getMae(isNeuralNetwork, r)[index])
                .min().orElse(0.0);
    }

    private double getMax(List<FoldResult> results, int index, boolean isNeuralNetwork) {
        return results.stream()
                .mapToDouble(r -> getMae(isNeuralNetwork, r)[index])
                .max().orElse(0.0);
    }

    private double getMinOverall(List<FoldResult> results, boolean isNeuralNetwork) {
        return results.stream()
                .mapToDouble(r -> getOverallMae(isNeuralNetwork, r))
                .min().orElse(0.0);
    }

    private double getMaxOverall(List<FoldResult> results, boolean isNeuralNetwork) {
        return results.stream()
                .mapToDouble(r -> getOverallMae(isNeuralNetwork, r))
                .max().orElse(0.0);
    }

    /**
     * Data structure for JSON serialization of predictions.
     */
    public static record PredictionData(
    		String testFamily,
			List<String> trainFamilies,
			Map<String, double[]> predictions,
			Map<String, double[]> actualLabels
	) {}
    
}