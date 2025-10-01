package uo.ml.neural.tracks.train.commands.lofo;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Handles evaluation of models against test datasets and baseline comparisons.
 */
public class ModelEvaluator {
    
    /**
     * Computes Mean Absolute Error between predictions and expectations.
     */
    public float[] computeMAE(INDArray predictions, INDArray expected) {
    	// predictions shape: [batchSize, 3]
    	// labels shape: [batchSize, 3]
        int batchSize = (int) predictions.size(0);
        float[] totalMAE = new float[3];
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 3; j++) {
                float pred = predictions.getFloat(i, j);
                float label = expected.getFloat(i, j);
                totalMAE[j] += Math.abs(pred - label);
            }
        }
        
        // Average over batch
        for (int j = 0; j < 3; j++) {
            totalMAE[j] /= batchSize;
        }
        
        return totalMAE;
    }
    
}