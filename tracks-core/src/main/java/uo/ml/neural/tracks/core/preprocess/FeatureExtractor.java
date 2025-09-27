package uo.ml.neural.tracks.core.preprocess;

import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.core.model.UtmPoint;

import java.util.ArrayList;
import java.util.List;

/**
 * Extracts geometric features from UTM point sequences for machine learning analysis.
 * Computes segment-based features including horizontal distance, vertical distance, and slope.
 */
public class FeatureExtractor {
    
    /**
     * Computes geometric features for each segment between consecutive UTM points.
     * 
     * @param pts List of UTM points (must contain at least 2 points)
     * @return List of segment features, with one less element than input points
     * @throws IllegalArgumentException if points list has fewer than 2 elements
     */
    public static List<SegmentFeature> computeFeatures(List<UtmPoint> pts) {
        if (pts.size() < 2) {
            throw new IllegalArgumentException("Need at least 2 points to compute features");
        }
        
        List<SegmentFeature> features = new ArrayList<>();
        
        for (int i = 1; i < pts.size(); i++) {
            UtmPoint prev = pts.get(i - 1);
            UtmPoint curr = pts.get(i);
            
            // Calculate 2D horizontal distance
            double de = curr.getE() - prev.getE();
            double dn = curr.getN() - prev.getN();
            double dh = Math.sqrt(de * de + dn * dn);
            
            // Calculate vertical distance (elevation change)
            double dz = curr.getZ() - prev.getZ();
            
            // Calculate slope with small epsilon to avoid division by zero
            double slope = dz / (dh + 1e-6);
            
            features.add(new SegmentFeature(dh, dz, slope));
        }
        
        return features;
    }
}