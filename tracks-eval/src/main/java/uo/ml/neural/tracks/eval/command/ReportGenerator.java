package uo.ml.neural.tracks.eval.command;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import uo.ml.neural.tracks.eval.model.LofoResult;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Generates evaluation reports in CSV and Markdown formats.
 * Creates statistical summaries with means and standard deviations.
 */
public class ReportGenerator {
    
    /**
     * Generates a CSV report with detailed LOFO results and aggregated statistics.
     */
    public void generateCSVReport(EvaluationResults results, Path outputPath) throws IOException {
        try (FileWriter writer = new FileWriter(outputPath.toFile());
             CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT)) {
            
            // Header for individual results
            printer.printRecord("Section", "Family", "Model", "Overall_MAE", "Distance_MAE", 
                               "Elevation_Pos_MAE", "Elevation_Neg_MAE", "Improvement_Pct");
            
            // Individual LOFO results
            List<LofoResult> lofoResults = results.getLofoResults();
            for (LofoResult result : lofoResults) {
                // Neural Network row
                printer.printRecord("Individual", result.getFamilyName(), "Neural_Network",
                    result.getNeuralNetworkMAE(),
                    result.getNeuralNetworkDetails()[0],
                    result.getNeuralNetworkDetails()[1], 
                    result.getNeuralNetworkDetails()[2],
                    "");
                
                // Baseline row
                printer.printRecord("Individual", result.getFamilyName(), "Baseline",
                    result.getBaselineMAE(),
                    result.getBaselineDetails()[0],
                    result.getBaselineDetails()[1],
                    result.getBaselineDetails()[2],
                    String.format("%.2f", result.getImprovementPercentage()));
            }
            
            // Empty row for separation
            printer.printRecord();
            
            // Aggregated statistics header
            printer.printRecord("Section", "Model", "Metric", "Mean", "Std_Dev", "Count", "", "");
            
            // Neural Network aggregated stats
            EvaluationResults.AggregatedStats nnStats = results.getNeuralNetworkStats();
            printer.printRecord("Aggregated", "Neural_Network", "Overall", 
                String.format("%.3f", nnStats.getOverallMean()),
                String.format("%.3f", nnStats.getOverallStd()),
                lofoResults.size());
            printer.printRecord("Aggregated", "Neural_Network", "Distance",
                String.format("%.3f", nnStats.getDistanceMean()),
                String.format("%.3f", nnStats.getDistanceStd()),
                lofoResults.size());
            printer.printRecord("Aggregated", "Neural_Network", "Elevation_Pos",
                String.format("%.3f", nnStats.getElevationPosMean()),
                String.format("%.3f", nnStats.getElevationPosStd()),
                lofoResults.size());
            printer.printRecord("Aggregated", "Neural_Network", "Elevation_Neg",
                String.format("%.3f", nnStats.getElevationNegMean()),
                String.format("%.3f", nnStats.getElevationNegStd()),
                lofoResults.size());
            
            // Baseline aggregated stats
            EvaluationResults.AggregatedStats baseStats = results.getBaselineStats();
            printer.printRecord("Aggregated", "Baseline", "Overall",
                String.format("%.3f", baseStats.getOverallMean()),
                String.format("%.3f", baseStats.getOverallStd()),
                lofoResults.size());
            printer.printRecord("Aggregated", "Baseline", "Distance",
                String.format("%.3f", baseStats.getDistanceMean()),
                String.format("%.3f", baseStats.getDistanceStd()),
                lofoResults.size());
            printer.printRecord("Aggregated", "Baseline", "Elevation_Pos",
                String.format("%.3f", baseStats.getElevationPosMean()),
                String.format("%.3f", baseStats.getElevationPosStd()),
                lofoResults.size());
            printer.printRecord("Aggregated", "Baseline", "Elevation_Neg",
                String.format("%.3f", baseStats.getElevationNegMean()),
                String.format("%.3f", baseStats.getElevationNegStd()),
                lofoResults.size());
        }
    }
    
    /**
     * Generates a Markdown report with formatted tables and statistics.
     */
    public void generateMarkdownReport(EvaluationResults results, Path outputPath) throws IOException {
        StringBuilder md = new StringBuilder();
        
        md.append("# GPS Tracks LOFO Evaluation Report\n\n");
        md.append("Generated on: ").append(java.time.LocalDateTime.now()).append("\n\n");
        
        // Individual Results Table
        md.append("## Individual LOFO Results\n\n");
        md.append("| Family | Neural Network | Baseline | Improvement | Improvement % |\n");
        md.append("|--------|----------------|----------|-------------|---------------|\n");
        
        List<LofoResult> lofoResults = results.getLofoResults();
        for (LofoResult result : lofoResults) {
            md.append(String.format("| %s | %.3f | %.3f | %.3f | %.1f%% |\n",
                result.getFamilyName(),
                result.getNeuralNetworkMAE(),
                result.getBaselineMAE(),
                result.getImprovement(),
                result.getImprovementPercentage()));
        }
        
        // Aggregated Statistics
        md.append("\n## Aggregated Statistics\n\n");
        md.append("### Neural Network Performance (Mean ± Std)\n\n");
        
        EvaluationResults.AggregatedStats nnStats = results.getNeuralNetworkStats();
        md.append("| Metric | Mean | Standard Deviation |\n");
        md.append("|--------|------|--------------------|\n");
        md.append(String.format("| Overall | %.3f | %.3f |\n", 
            nnStats.getOverallMean(), nnStats.getOverallStd()));
        md.append(String.format("| Distance | %.3f | %.3f |\n",
            nnStats.getDistanceMean(), nnStats.getDistanceStd()));
        md.append(String.format("| Elevation (+) | %.3f | %.3f |\n",
            nnStats.getElevationPosMean(), nnStats.getElevationPosStd()));
        md.append(String.format("| Elevation (-) | %.3f | %.3f |\n",
            nnStats.getElevationNegMean(), nnStats.getElevationNegStd()));
        
        md.append("\n### Baseline Performance (Mean ± Std)\n\n");
        EvaluationResults.AggregatedStats baseStats = results.getBaselineStats();
        md.append("| Metric | Mean | Standard Deviation |\n");
        md.append("|--------|------|--------------------|\n");
        md.append(String.format("| Overall | %.3f | %.3f |\n",
            baseStats.getOverallMean(), baseStats.getOverallStd()));
        md.append(String.format("| Distance | %.3f | %.3f |\n",
            baseStats.getDistanceMean(), baseStats.getDistanceStd()));
        md.append(String.format("| Elevation (+) | %.3f | %.3f |\n",
            baseStats.getElevationPosMean(), baseStats.getElevationPosStd()));
        md.append(String.format("| Elevation (-) | %.3f | %.3f |\n",
            baseStats.getElevationNegMean(), baseStats.getElevationNegStd()));
        
        // Summary Analysis
        md.append("\n## Summary Analysis\n\n");
        double overallImprovement = baseStats.getOverallMean() - nnStats.getOverallMean();
        double improvementPct = (overallImprovement / baseStats.getOverallMean()) * 100;
        
        md.append(String.format("- **Overall Improvement**: %.3f MAE reduction (%.1f%% better)\n",
            overallImprovement, improvementPct));
        md.append(String.format("- **Total LOFO Folds**: %d\n", lofoResults.size()));
        
        int betterCount = (int) lofoResults.stream()
            .mapToDouble(LofoResult::getImprovement)
            .filter(imp -> imp > 0)
            .count();
        
        md.append(String.format("- **Folds where NN outperformed baseline**: %d/%d (%.1f%%)\n",
            betterCount, lofoResults.size(), (betterCount * 100.0) / lofoResults.size()));
        
        Files.writeString(outputPath, md.toString());
    }
    
    /**
     * Generates a summary text report with key findings.
     */
    public void generateSummaryReport(EvaluationResults results, Path outputPath) throws IOException {
        StringBuilder summary = new StringBuilder();
        
        summary.append("GPS TRACKS EVALUATION SUMMARY\n");
        summary.append("============================\n\n");
        
        List<LofoResult> lofoResults = results.getLofoResults();
        EvaluationResults.AggregatedStats nnStats = results.getNeuralNetworkStats();
        EvaluationResults.AggregatedStats baseStats = results.getBaselineStats();
        
        summary.append(String.format("Total LOFO Folds: %d\n", lofoResults.size()));
        summary.append(String.format("Neural Network MAE: %.3f ± %.3f\n",
            nnStats.getOverallMean(), nnStats.getOverallStd()));
        summary.append(String.format("Baseline MAE: %.3f ± %.3f\n",
            baseStats.getOverallMean(), baseStats.getOverallStd()));
        
        double improvement = baseStats.getOverallMean() - nnStats.getOverallMean();
        double improvementPct = (improvement / baseStats.getOverallMean()) * 100;
        
        summary.append(String.format("\nImprovement: %.3f MAE (%.1f%% better)\n", improvement, improvementPct));
        
        // Best and worst performing families
        LofoResult bestFamily = lofoResults.stream()
            .max((a, b) -> Double.compare(a.getImprovement(), b.getImprovement()))
            .orElse(null);
        
        LofoResult worstFamily = lofoResults.stream()
            .min((a, b) -> Double.compare(a.getImprovement(), b.getImprovement()))
            .orElse(null);
        
        if (bestFamily != null) {
            summary.append(String.format("\nBest performing family: %s (%.1f%% improvement)\n",
                bestFamily.getFamilyName(), bestFamily.getImprovementPercentage()));
        }
        
        if (worstFamily != null) {
            summary.append(String.format("Worst performing family: %s (%.1f%% improvement)\n",
                worstFamily.getFamilyName(), worstFamily.getImprovementPercentage()));
        }
        
        Files.writeString(outputPath, summary.toString());
    }
}