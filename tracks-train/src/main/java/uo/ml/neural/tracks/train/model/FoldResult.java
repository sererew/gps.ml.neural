package uo.ml.neural.tracks.train.model;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Represents the result of a single fold in LOFO cross-validation.
 */
public class FoldResult {
    private final String testFamily;
    private final List<String> trainFamilies;
    private final double[] nnMAE;
    private final double nnOverallMAE;
    private final double[] baselineMAE;
    private final double baselineOverallMAE;
    private final MultiLayerNetwork trainedModel;
    private final Map<String, double[]> predictions;
    private final Map<String, double[]> actualLabels;
    
    public FoldResult(String testFamily, 
                     List<String> trainFamilies,
                     double[] nnMAE, double nnOverallMAE, 
                     double[] baselineMAE, double baselineOverallMAE,
                     MultiLayerNetwork trainedModel,
                     Map<String, double[]> predictions,
                     Map<String, double[]> actualLabels) {
        
        this.testFamily = testFamily;
        this.trainFamilies = List.copyOf(trainFamilies);
        this.nnMAE = nnMAE.clone();
        this.nnOverallMAE = nnOverallMAE;
        this.baselineMAE = baselineMAE.clone();
        this.baselineOverallMAE = baselineOverallMAE;
        this.trainedModel = trainedModel;
        this.predictions = Map.copyOf(predictions);
        this.actualLabels = Map.copyOf(actualLabels);
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
    
    public List<String> getTrainFamilies() {
        return trainFamilies;
    }
    
    public MultiLayerNetwork getTrainedModel() {
        return trainedModel;
    }
    
    public Map<String, double[]> getPredictions() {
        return predictions;
    }
    
    public Map<String, double[]> getActualLabels() {
        return actualLabels;
    }
}