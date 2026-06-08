package uo.ml.neural.tracks.eval.command;

import java.util.List;

import uo.ml.neural.tracks.eval.model.BaselineResult;
import uo.ml.neural.tracks.eval.model.LofoResult;

/**
 * Container for all evaluation results including LOFO and baseline comparisons.
 */
public class EvaluationResults {
    
    private final List<LofoResult> lofoResults;
    private final List<BaselineResult> baselineResults;
    
    public EvaluationResults(List<LofoResult> lofoResults, List<BaselineResult> baselineResults) {
        this.lofoResults = lofoResults;
        this.baselineResults = baselineResults;
    }
    
    public List<LofoResult> getLofoResults() {
        return lofoResults;
    }
    
    public List<BaselineResult> getBaselineResults() {
        return baselineResults;
    }
    
    /**
     * Calculates aggregated statistics for neural network results.
     */
    public AggregatedStats getNeuralNetworkStats() {
        if (lofoResults.isEmpty()) {
            return new AggregatedStats(0, 0, 0, 0, 0, 0, 0, 0);
        }
        
        double[] overallMAEs = lofoResults.stream()
            .mapToDouble(LofoResult::getNeuralNetworkMAE)
            .toArray();
        
        double[] distMAEs = lofoResults.stream()
            .mapToDouble(r -> r.getNeuralNetworkDetails()[0])
            .toArray();
        
        double[] posMAEs = lofoResults.stream()
            .mapToDouble(r -> r.getNeuralNetworkDetails()[1])
            .toArray();
        
        double[] negMAEs = lofoResults.stream()
            .mapToDouble(r -> r.getNeuralNetworkDetails()[2])
            .toArray();
        
        return new AggregatedStats(
            calculateMean(overallMAEs), calculateStd(overallMAEs),
            calculateMean(distMAEs), calculateStd(distMAEs),
            calculateMean(posMAEs), calculateStd(posMAEs),
            calculateMean(negMAEs), calculateStd(negMAEs)
        );
    }
    
    /**
     * Calculates aggregated statistics for baseline results.
     */
    public AggregatedStats getBaselineStats() {
        if (lofoResults.isEmpty()) {
            return new AggregatedStats(0, 0, 0, 0, 0, 0, 0, 0);
        }
        
        double[] overallMAEs = lofoResults.stream()
            .mapToDouble(LofoResult::getBaselineMAE)
            .toArray();
        
        double[] distMAEs = lofoResults.stream()
            .mapToDouble(r -> r.getBaselineDetails()[0])
            .toArray();
        
        double[] posMAEs = lofoResults.stream()
            .mapToDouble(r -> r.getBaselineDetails()[1])
            .toArray();
        
        double[] negMAEs = lofoResults.stream()
            .mapToDouble(r -> r.getBaselineDetails()[2])
            .toArray();
        
        return new AggregatedStats(
            calculateMean(overallMAEs), calculateStd(overallMAEs),
            calculateMean(distMAEs), calculateStd(distMAEs),
            calculateMean(posMAEs), calculateStd(posMAEs),
            calculateMean(negMAEs), calculateStd(negMAEs)
        );
    }
    
    private double calculateMean(double[] values) {
        if (values.length == 0) return 0.0;
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    
    private double calculateStd(double[] values) {
        if (values.length <= 1) return 0.0;
        
        double mean = calculateMean(values);
        double sumSquaredDiffs = 0.0;
        
        for (double value : values) {
            sumSquaredDiffs += Math.pow(value - mean, 2);
        }
        
        return Math.sqrt(sumSquaredDiffs / values.length);
    }
    
    /**
     * Represents aggregated statistics for a model type.
     */
    public static class AggregatedStats {
        private final double overallMean, overallStd;
        private final double distanceMean, distanceStd;
        private final double elevationPosMean, elevationPosStd;
        private final double elevationNegMean, elevationNegStd;
        
        public AggregatedStats(double overallMean, double overallStd,
                              double distanceMean, double distanceStd,
                              double elevationPosMean, double elevationPosStd,
                              double elevationNegMean, double elevationNegStd) {
            this.overallMean = overallMean;
            this.overallStd = overallStd;
            this.distanceMean = distanceMean;
            this.distanceStd = distanceStd;
            this.elevationPosMean = elevationPosMean;
            this.elevationPosStd = elevationPosStd;
            this.elevationNegMean = elevationNegMean;
            this.elevationNegStd = elevationNegStd;
        }
        
        // Getters
        public double getOverallMean() { return overallMean; }
        public double getOverallStd() { return overallStd; }
        public double getDistanceMean() { return distanceMean; }
        public double getDistanceStd() { return distanceStd; }
        public double getElevationPosMean() { return elevationPosMean; }
        public double getElevationPosStd() { return elevationPosStd; }
        public double getElevationNegMean() { return elevationNegMean; }
        public double getElevationNegStd() { return elevationNegStd; }
    }
}