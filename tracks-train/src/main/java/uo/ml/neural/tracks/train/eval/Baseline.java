package uo.ml.neural.tracks.train.eval;

import uo.ml.neural.tracks.core.model.SegmentFeature;

import java.util.List;

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
    public static double[] computeBaseline(List<SegmentFeature> features) {
        if (features.isEmpty()) {
            return new double[]{0.0, 0.0, 0.0};
        }
        
        double totalDistance = 0.0;
        double positiveElevation = 0.0;
        double negativeElevation = 0.0;
        
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
        
        return new double[]{totalDistance, positiveElevation, negativeElevation};
    }
    
    /**
     * Computes Mean Absolute Error between baseline prediction and true labels.
     * 
     * @param prediction Baseline prediction [dist, desn_pos, desn_neg]
     * @param trueLabels True labels [dist_total, desn_pos, desn_neg] 
     * @return MAE for each component [mae_dist, mae_pos, mae_neg]
     */
    public static double[] computeMAE(double[] prediction, double[] trueLabels) {
        if (prediction.length != 3 || trueLabels.length != 3) {
            throw new IllegalArgumentException("Both prediction and labels must have 3 components");
        }
        
        double[] mae = new double[3];
        for (int i = 0; i < 3; i++) {
            mae[i] = Math.abs(prediction[i] - trueLabels[i]);
        }
        
        return mae;
    }
    
    /**
     * Computes overall MAE as the mean of individual component MAEs.
     * 
     * @param prediction Baseline prediction
     * @param trueLabels True labels
     * @return Overall MAE (mean of the 3 component MAEs)
     */
    public static double computeOverallMAE(double[] prediction, double[] trueLabels) {
        double[] mae = computeMAE(prediction, trueLabels);
        return (mae[0] + mae[1] + mae[2]) / 3.0;
    }
}