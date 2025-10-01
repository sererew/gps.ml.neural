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
    private final float[] nnMAE;
    private final float nnOverallMAE;
    private final MultiLayerNetwork trainedModel;
    private final Map<String, float[]> predictions;
    private final Map<String, float[]> actualLabels;
    
    public FoldResult(String testFamily, 
                     List<String> trainFamilies,
                     float[] nnMAE, float nnOverallMAE, 
                     MultiLayerNetwork trainedModel,
                     Map<String, float[]> predictions,
                     Map<String, float[]> actualLabels) {
        
        this.testFamily = testFamily;
        this.trainFamilies = List.copyOf(trainFamilies);
        this.nnMAE = nnMAE.clone();
        this.nnOverallMAE = nnOverallMAE;
        this.trainedModel = trainedModel;
        this.predictions = Map.copyOf(predictions);
        this.actualLabels = Map.copyOf(actualLabels);
    }
    
    public String getTestFamily() {
        return testFamily;
    }
    
    public float[] getNnMAE() {
        return nnMAE.clone();
    }
    
    public float getNnOverallMAE() {
        return nnOverallMAE;
    }
    
    public List<String> getTrainFamilies() {
        return trainFamilies;
    }
    
    public MultiLayerNetwork getTrainedModel() {
        return trainedModel;
    }
    
    public Map<String, float[]> getPredictions() {
        return predictions;
    }
    
    public Map<String, float[]> getActualLabels() {
        return actualLabels;
    }
}