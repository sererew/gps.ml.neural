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
    public float[] computeMAE(INDArray predictions, INDArray labels) {
        int batchSize = (int) predictions.size(0);
        float[] totalMAE = new float[3];
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 3; j++) {
                float pred = predictions.getFloat(i, j);
                float label = labels.getFloat(i, j);
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
    public float[] evaluateBaseline(SequenceDataset testData) {
        int batchSize = testData.getBatchSize();
        float[] totalMAE = new float[3];
        
        for (int i = 0; i < batchSize; i++) {
            // Extract features for this sample
            List<SegmentFeature> features = toFeaturesList(testData.getFeatures(), i);
            
            // Compute baseline prediction
            float[] expected = Baseline.computeBaseline(features);
            
            // Get true labels, the inferred values by the model
            INDArray labels = testData.getLabels();
			float[] prediction = new float[]{
                labels.getFloat(i, 0),
                labels.getFloat(i, 1), 
                labels.getFloat(i, 2)
            };
            
            // Compute Absolute Error for this sample
            float[] sampleAE = Baseline.absError(prediction, expected);
            for (int j = 0; j < 3; j++) {
                totalMAE[j] += sampleAE[j];
            }
        }
        
        // Average over batch
        for (int j = 0; j < 3; j++) {
            totalMAE[j] /= batchSize;
        }
        
        return totalMAE;
    }
    
    private List<SegmentFeature> toFeaturesList(INDArray features, int sampleIndex) {
        List<SegmentFeature> result = new ArrayList<>();
        int maxLength = (int) features.size(2);
        
        for (int t = 0; t < maxLength; t++) {
            float dh = features.getFloat(sampleIndex, 0, t);
            float dz = features.getFloat(sampleIndex, 1, t);
            float slope = features.getFloat(sampleIndex, 2, t);
            
            // Stop at padding (assuming zero-padding)
            if (dh == 0.0 && dz == 0.0 && slope == 0.0) {
                break;
            }
            
            result.add(new SegmentFeature(dh, dz, slope));
        }
        
        return result;
    }
}