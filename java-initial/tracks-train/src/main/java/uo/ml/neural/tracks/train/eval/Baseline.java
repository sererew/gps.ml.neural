package uo.ml.neural.tracks.train.eval;

import java.util.List;

import uo.ml.neural.tracks.core.model.SegmentFeature;

/**
 * Baseline evaluator that computes simple geometric statistics from GPS tracks.
 * Provides a baseline comparison for machine learning models by calculating
 * basic distance and elevation metrics directly from track features.
 */
public class Baseline {
    
    /**
     * Computes baseline predictions for a track given its segment features.
     * 
     * @param features List of segment features (dh, dz, slope) from a track
     * @return Array with [total_distance, positive_elevation, negative_elevation]
     */
    public static float[] computeBaseline(List<SegmentFeature> features) {
        if (features.isEmpty()) {
            return new float[]{0.0f, 0.0f, 0.0f};
        }
        
        float totalDistance = 0.0f;
        float positiveElevation = 0.0f;
        float negativeElevation = 0.0f;
        
        for (SegmentFeature feature : features) {
            // Total horizontal distance: sum all dh values
            totalDistance += feature.getDh();
            
            // Positive elevation gain: sum all positive dz values
            if (feature.getDz() > 0) {
                positiveElevation += feature.getDz();
            }
            
            // Negative elevation loss: sum absolute values of negative dz
            if (feature.getDz() < 0) {
                negativeElevation += Math.abs(feature.getDz());
            }
        }
        
        return new float[]{totalDistance, positiveElevation, negativeElevation};
    }
    
    /**
     * Computes Absolute Error between baseline prediction and true labels.
     * 
     * @param prediction Baseline prediction [dist, desn_pos, desn_neg]
     * @param expected True labels [dist_total, desn_pos, desn_neg] 
     * @return MAE for each component [mae_dist, mae_pos, mae_neg]
     */
    public static float[] absError(float[] prediction, float[] expected) {
        if (prediction.length != 3 || expected.length != 3) {
            throw new IllegalArgumentException("Both prediction and labels "
            		+ "must have 3 components");
        }
        
        float[] mae = new float[3];
        for (int i = 0; i < 3; i++) {
            mae[i] = Math.abs(prediction[i] - expected[i]);
        }
        
        return mae;
    }
    
    /**
     * Computes Mean Absolute Error as the mean of individual absolute errors.
     * 
     * @param prediction Baseline prediction
     * @param trueLabels True labels
     * @return Overall MAE (mean of the 3 component MAEs)
     */
    public static float computeOverallMAE(float[] prediction, float[] trueLabels) {
        float[] mae = absError(prediction, trueLabels);
        return (mae[0] + mae[1] + mae[2]) / 3.0f;
    }
}