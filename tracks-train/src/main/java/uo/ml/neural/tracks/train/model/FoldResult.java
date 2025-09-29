package uo.ml.neural.tracks.train.model;

/**
 * Represents the result of a single fold in LOFO cross-validation.
 */
public class FoldResult {
    private final String testFamily;
    private final double[] nnMAE;
    private final double nnOverallMAE;
    private final double[] baselineMAE;
    private final double baselineOverallMAE;
    
    public FoldResult(String testFamily, double[] nnMAE, double nnOverallMAE, 
                     double[] baselineMAE, double baselineOverallMAE) {
        this.testFamily = testFamily;
        this.nnMAE = nnMAE.clone();
        this.nnOverallMAE = nnOverallMAE;
        this.baselineMAE = baselineMAE.clone();
        this.baselineOverallMAE = baselineOverallMAE;
    }
    
    public String getTestFamily() {
        return testFamily;
    }
    
    public double[] getNnMAE() {
        return nnMAE.clone();
    }
    
    public double getNnOverallMAE() {
        return nnOverallMAE;
    }
    
    public double[] getBaselineMAE() {
        return baselineMAE.clone();
    }
    
    public double getBaselineOverallMAE() {
        return baselineOverallMAE;
    }
}