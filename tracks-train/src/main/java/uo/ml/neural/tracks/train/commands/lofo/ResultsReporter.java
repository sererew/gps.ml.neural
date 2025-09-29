package uo.ml.neural.tracks.train.commands.lofo;

import java.util.List;

import uo.ml.neural.tracks.train.model.FoldResult;

/**
 * Handles the generation of statistical reports from LOFO validation results.
 */
public class ResultsReporter {
    
    /**
     * Reports overall results from all folds with means and standard deviations.
     */
    public void reportOverallResults(List<FoldResult> results) {
        System.out.println("=== OVERALL RESULTS ===");
        
        // Compute means and standard deviations
        double[] nnMeans = new double[4]; // 3 components + overall
        double[] nnStds = new double[4];
        double[] baselineMeans = new double[4];
        double[] baselineStds = new double[4];
        
        computeMeans(results, nnMeans, baselineMeans);
        computeStandardDeviations(results, nnMeans, baselineMeans, nnStds, baselineStds);
        
        printStatistics(nnMeans, nnStds, baselineMeans, baselineStds);
        printIndividualResults(results);
    }
    
    private void computeMeans(List<FoldResult> results, double[] nnMeans, double[] baselineMeans) {
        for (FoldResult result : results) {
            double[] nnMAE = result.getNnMAE();
            double[] baselineMAE = result.getBaselineMAE();
            
            for (int i = 0; i < 3; i++) {
                nnMeans[i] += nnMAE[i];
                baselineMeans[i] += baselineMAE[i];
            }
            nnMeans[3] += result.getNnOverallMAE();
            baselineMeans[3] += result.getBaselineOverallMAE();
        }
        
        for (int i = 0; i < 4; i++) {
            nnMeans[i] /= results.size();
            baselineMeans[i] /= results.size();
        }
    }
    
    private void computeStandardDeviations(List<FoldResult> results, double[] nnMeans, 
                                         double[] baselineMeans, double[] nnStds, double[] baselineStds) {
        for (FoldResult result : results) {
            double[] nnMAE = result.getNnMAE();
            double[] baselineMAE = result.getBaselineMAE();
            
            for (int i = 0; i < 3; i++) {
                nnStds[i] += Math.pow(nnMAE[i] - nnMeans[i], 2);
                baselineStds[i] += Math.pow(baselineMAE[i] - baselineMeans[i], 2);
            }
            nnStds[3] += Math.pow(result.getNnOverallMAE() - nnMeans[3], 2);
            baselineStds[3] += Math.pow(result.getBaselineOverallMAE() - baselineMeans[3], 2);
        }
        
        for (int i = 0; i < 4; i++) {
            nnStds[i] = Math.sqrt(nnStds[i] / results.size());
            baselineStds[i] = Math.sqrt(baselineStds[i] / results.size());
        }
    }
    
    private void printStatistics(double[] nnMeans, double[] nnStds, 
                               double[] baselineMeans, double[] baselineStds) {
        System.out.printf("Neural Network MAE (mean ± std):%n");
        System.out.printf("  Distance:  %.3f ± %.3f%n", nnMeans[0], nnStds[0]);
        System.out.printf("  Elev (+):  %.3f ± %.3f%n", nnMeans[1], nnStds[1]);
        System.out.printf("  Elev (-):  %.3f ± %.3f%n", nnMeans[2], nnStds[2]);
        System.out.printf("  Overall:   %.3f ± %.3f%n", nnMeans[3], nnStds[3]);
        System.out.println();
        
        System.out.printf("Baseline MAE (mean ± std):%n");
        System.out.printf("  Distance:  %.3f ± %.3f%n", baselineMeans[0], baselineStds[0]);
        System.out.printf("  Elev (+):  %.3f ± %.3f%n", baselineMeans[1], baselineStds[1]);
        System.out.printf("  Elev (-):  %.3f ± %.3f%n", baselineMeans[2], baselineStds[2]);
        System.out.printf("  Overall:   %.3f ± %.3f%n", baselineMeans[3], baselineStds[3]);
        System.out.println();
    }
    
    private void printIndividualResults(List<FoldResult> results) {
        System.out.println("Results by family:");
        System.out.println("Family\t\tNeural Network\t\tBaseline");
        System.out.println("------\t\t--------------\t\t--------");
        for (FoldResult result : results) {
            System.out.printf("%-12s\t%.3f\t\t\t%.3f%n", 
                result.getTestFamily(), result.getNnOverallMAE(), result.getBaselineOverallMAE());
        }
    }
}