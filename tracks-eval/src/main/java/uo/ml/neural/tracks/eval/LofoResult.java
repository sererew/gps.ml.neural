package uo.ml.neural.tracks.eval;

/**
 * Represents the results of a single LOFO fold, comparing neural network vs baseline performance.
 */
public class LofoResult {
    
    private final String familyName;
    private final double neuralNetworkMAE;
    private final double baselineMAE;
    private final double[] neuralNetworkDetails; // [distance, elevation_pos, elevation_neg]
    private final double[] baselineDetails;      // [distance, elevation_pos, elevation_neg]
    
    public LofoResult(String familyName, double neuralNetworkMAE, double baselineMAE,
                      double[] neuralNetworkDetails, double[] baselineDetails) {
        this.familyName = familyName;
        this.neuralNetworkMAE = neuralNetworkMAE;
        this.baselineMAE = baselineMAE;
        this.neuralNetworkDetails = neuralNetworkDetails.clone();
        this.baselineDetails = baselineDetails.clone();
    }
    
    public String getFamilyName() {
        return familyName;
    }
    
    public double getNeuralNetworkMAE() {
        return neuralNetworkMAE;
    }
    
    public double getBaselineMAE() {
        return baselineMAE;
    }
    
    public double[] getNeuralNetworkDetails() {
        return neuralNetworkDetails.clone();
    }
    
    public double[] getBaselineDetails() {
        return baselineDetails.clone();
    }
    
    /**
     * Calculates the improvement of neural network over baseline.
     * Positive values indicate neural network is better (lower MAE).
     */
    public double getImprovement() {
        return baselineMAE - neuralNetworkMAE;
    }
    
    /**
     * Calculates the improvement percentage.
     */
    public double getImprovementPercentage() {
        if (baselineMAE == 0) return 0;
        return (getImprovement() / baselineMAE) * 100;
    }
    
    @Override
    public String toString() {
        return String.format("LofoResult{family='%s', NN=%.3f, Baseline=%.3f, Improvement=%.1f%%}",
            familyName, neuralNetworkMAE, baselineMAE, getImprovementPercentage());
    }
}