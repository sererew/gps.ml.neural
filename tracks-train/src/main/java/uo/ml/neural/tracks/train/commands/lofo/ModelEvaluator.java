package uo.ml.neural.tracks.train.commands.lofo;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.train.data.SequenceDataset;
import uo.ml.neural.tracks.train.eval.Baseline;

/**
 * Handles evaluation of models against test datasets and baseline comparisons.
 */
public class ModelEvaluator {
    
    /**
     * Computes Mean Absolute Error between predictions and labels.
     */
    public double[] computeMAE(INDArray predictions, INDArray labels) {
        int batchSize = (int) predictions.size(0);
        double[] totalMAE = new double[3];
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 3; j++) {
                double pred = predictions.getDouble(i, j);
                double label = labels.getDouble(i, j);
                totalMAE[j] += Math.abs(pred - label);
            }
        }
        
        // Average over batch
        for (int j = 0; j < 3; j++) {
            totalMAE[j] /= batchSize;
        }
        
        return totalMAE;
    }
    
    /**
     * Evaluates baseline model on test dataset.
     */
    public double[] evaluateBaseline(SequenceDataset testData) {
        int batchSize = testData.getBatchSize();
        double[] totalMAE = new double[3];
        
        for (int i = 0; i < batchSize; i++) {
            // Extract features for this sample
            List<SegmentFeature> features = extractFeaturesFromArray(testData.getFeatures(), i);
            
            // Compute baseline prediction
            double[] baseline = Baseline.computeBaseline(features);
            
            // Get true labels
            double[] expectedValues = new double[]{
                testData.getLabels().getDouble(i, 0),
                testData.getLabels().getDouble(i, 1), 
                testData.getLabels().getDouble(i, 2)
            };
            
            // Compute MAE for this sample
            double[] sampleMAE = Baseline.computeMAE(baseline, expectedValues);
            for (int j = 0; j < 3; j++) {
                totalMAE[j] += sampleMAE[j];
            }
        }
        
        // Average over batch
        for (int j = 0; j < 3; j++) {
            totalMAE[j] /= batchSize;
        }
        
        return totalMAE;
    }
    
    private List<SegmentFeature> extractFeaturesFromArray(INDArray features, int sampleIndex) {
        List<SegmentFeature> result = new ArrayList<>();
        int maxLength = (int) features.size(2);
        
        for (int t = 0; t < maxLength; t++) {
            double dh = features.getDouble(sampleIndex, 0, t);
            double dz = features.getDouble(sampleIndex, 1, t);
            double slope = features.getDouble(sampleIndex, 2, t);
            
            // Stop at padding (assuming zero-padding)
            if (dh == 0.0 && dz == 0.0 && slope == 0.0) {
                break;
            }
            
            result.add(new SegmentFeature(dh, dz, slope));
        }
        
        return result;
    }
}