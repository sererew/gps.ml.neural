package uo.ml.neural.tracks.core.preprocess;

import uo.ml.neural.tracks.core.model.SegmentFeature;

import java.util.List;

/**
 * Utility class for padding feature sequences to uniform length.
 * Essential for batch processing in machine learning models that require fixed-size inputs.
 */
public class Padding {
    
    /**
     * Pads a sequence of segment features to a specified maximum length.
     * Features are placed at the beginning, and remaining positions are filled with zeros.
     * 
     * @param seq List of segment features to pad
     * @param maxLen Maximum length for the output array
     * @return 2D array with shape [maxLen][3] containing [dh, dz, slope] values
     * @throws IllegalArgumentException if maxLen is not positive or sequence is longer than maxLen
     */
    public static double[][] padFeatures(List<SegmentFeature> seq, int maxLen) {
        if (maxLen <= 0) {
            throw new IllegalArgumentException("Maximum length must be positive");
        }
        if (seq.size() > maxLen) {
            throw new IllegalArgumentException("Sequence length (" + seq.size() + ") exceeds maximum length (" + maxLen + ")");
        }
        
        double[][] padded = new double[maxLen][3];
        
        // Fill with actual feature values
        for (int i = 0; i < seq.size(); i++) {
            SegmentFeature feature = seq.get(i);
            padded[i][0] = feature.getDh();
            padded[i][1] = feature.getDz();
            padded[i][2] = feature.getSlope();
        }
        
        // Remaining positions are already initialized to 0.0 by default
        
        return padded;
    }
    
    /**
     * Creates a mask array indicating which positions contain real data vs padding.
     * 
     * @param realLen Actual length of the sequence (number of real elements)
     * @param maxLen Maximum length (total array size)
     * @return Mask array where 1.0 indicates real data and 0.0 indicates padding
     * @throws IllegalArgumentException if realLen > maxLen or either parameter is negative
     */
    public static double[] makeMask(int realLen, int maxLen) {
        if (realLen < 0 || maxLen < 0) {
            throw new IllegalArgumentException("Lengths must be non-negative");
        }
        if (realLen > maxLen) {
            throw new IllegalArgumentException("Real length (" + realLen + ") cannot exceed maximum length (" + maxLen + ")");
        }
        
        double[] mask = new double[maxLen];
        
        // Set 1.0 for real data positions
        for (int i = 0; i < realLen; i++) {
            mask[i] = 1.0;
        }
        
        // Remaining positions are already 0.0 by default
        
        return mask;
    }
}