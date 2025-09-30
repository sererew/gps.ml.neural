package uo.ml.neural.tracks.train.commands.lofo;

import java.util.List;

import uo.ml.neural.tracks.train.model.FoldResult;

/**
 * Handles console reporting of LOFO cross-validation results.
 */
public class ResultsReporter {

    /**
     * Reports overall LOFO cross-validation results to console.
     */
    public void reportOverallResults(List<FoldResult> foldResults) {
        if (foldResults.isEmpty()) {
            System.out.println("No fold results to report.");
            return;
        }

        System.out.println("\n" + "=".repeat(60));
        System.out.println("LOFO CROSS-VALIDATION SUMMARY");
        System.out.println("=".repeat(60));

        // Calculate overall statistics
        double[] nnMeans = calculateMeans(foldResults, true);
        double[] nnStds = calculateStds(foldResults, nnMeans, true);
        double[] baselineMeans = calculateMeans(foldResults, false);
        double[] baselineStds = calculateStds(foldResults, baselineMeans, false);

        System.out.printf("Total folds: %d%n%n", foldResults.size());

        // Neural Network Results
        System.out.println("NEURAL NETWORK PERFORMANCE:");
        System.out.println("-".repeat(40));
        System.out.printf("Distance MAE:     %.3f ± %.3f%n", nnMeans[0], nnStds[0]);
        System.out.printf("Elevation+ MAE:   %.3f ± %.3f%n", nnMeans[1], nnStds[1]);
        System.out.printf("Elevation- MAE:   %.3f ± %.3f%n", nnMeans[2], nnStds[2]);
        System.out.printf("Overall MAE:      %.3f ± %.3f%n%n", nnMeans[3], nnStds[3]);

        // Baseline Results
        System.out.println("BASELINE PERFORMANCE:");
        System.out.println("-".repeat(40));
        System.out.printf("Distance MAE:     %.3f ± %.3f%n", baselineMeans[0], baselineStds[0]);
        System.out.printf("Elevation+ MAE:   %.3f ± %.3f%n", baselineMeans[1], baselineStds[1]);
        System.out.printf("Elevation- MAE:   %.3f ± %.3f%n", baselineMeans[2], baselineStds[2]);
        System.out.printf("Overall MAE:      %.3f ± %.3f%n%n", baselineMeans[3], baselineStds[3]);

        // Improvement analysis
        double overallImprovement = ((baselineMeans[3] - nnMeans[3]) / baselineMeans[3]) * 100;
        System.out.printf("Overall improvement: %.1f%%%n", overallImprovement);
        System.out.println("=".repeat(60));
    }

    private double[] calculateMeans(List<FoldResult> results, boolean isNeuralNetwork) {
        double[] sums = new double[4]; // 3 MAE values + overall
        
        for (FoldResult result : results) {
            double[] mae = isNeuralNetwork 
            		? result.getNnMAE() 
            		: result.getBaselineMAE();
            double overall = isNeuralNetwork 
            		? result.getNnOverallMAE() 
            		: result.getBaselineOverallMAE();
            
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

    private double[] calculateStds(
    		List<FoldResult> results, 
    		double[] means, 
    		boolean isNeuralNetwork) {
    	
        double[] variances = new double[4];
        
        for (FoldResult result : results) {
            double[] mae = isNeuralNetwork 
            		? result.getNnMAE() 
            		: result.getBaselineMAE();
            double overall = isNeuralNetwork 
            		? result.getNnOverallMAE() 
            		: result.getBaselineOverallMAE();
            
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
}