package uo.ml.neural.tracks.core.preprocess;

import uo.ml.neural.tracks.core.model.UtmPoint;

import java.util.ArrayList;
import java.util.List;

/**
 * 3D resampler for GPS tracks that interpolates points based on 3D arc length.
 * This ensures uniform spacing along the track considering both horizontal and vertical movement.
 */
public class Resampler3D {
    
    /**
     * Resamples a track by 3D arc length with uniform step size.
     * The first point is always included, then points are interpolated at regular 3D distances.
     * 
     * @param pts Original list of UTM points
     * @param stepMeters Distance between resampled points in meters (3D distance)
     * @return List of resampled UTM points with uniform 3D spacing
     * @throws IllegalArgumentException if points list is empty or stepMeters is not positive
     */
    public static List<UtmPoint> resampleByArcLength3D(List<UtmPoint> pts, double stepMeters) {
        if (pts.isEmpty()) {
            throw new IllegalArgumentException("Points list cannot be empty");
        }
        if (stepMeters <= 0) {
            throw new IllegalArgumentException("Step size must be positive");
        }
        if (pts.size() == 1) {
            return new ArrayList<>(pts);
        }
        
        List<UtmPoint> resampled = new ArrayList<>();
        resampled.add(pts.get(0)); // Always include first point
        
        double targetDistance = stepMeters;
        double totalDistance = 0.0;
        
        for (int i = 1; i < pts.size(); i++) {
            UtmPoint prev = pts.get(i - 1);
            UtmPoint curr = pts.get(i);
            
            double segmentDistance = distance3D(prev, curr);
            double segmentStart = totalDistance;
            double segmentEnd = totalDistance + segmentDistance;
            
            // Check if we need to interpolate points in this segment
            while (targetDistance <= segmentEnd) {
                double distanceInSegment = targetDistance - segmentStart;
                double ratio = segmentDistance > 0 ? distanceInSegment / segmentDistance : 0.0;
                
                // Interpolate point at target distance
                UtmPoint interpolated = interpolate(prev, curr, ratio);
                resampled.add(interpolated);
                
                // Update for next target
                targetDistance += stepMeters;
            }
            
            totalDistance = segmentEnd;
        }
        
        return resampled;
    }
    
    /**
     * Calculates the 3D Euclidean distance between two UTM points.
     * 
     * @param p1 First point
     * @param p2 Second point
     * @return 3D distance in meters
     */
    private static double distance3D(UtmPoint p1, UtmPoint p2) {
        double de = p2.getE() - p1.getE();
        double dn = p2.getN() - p1.getN();
        double dz = p2.getZ() - p1.getZ();
        
        return Math.sqrt(de * de + dn * dn + dz * dz);
    }
    
    /**
     * Linearly interpolates between two UTM points.
     * 
     * @param p1 Start point
     * @param p2 End point
     * @param ratio Interpolation ratio (0.0 = p1, 1.0 = p2)
     * @return Interpolated UTM point
     */
    private static UtmPoint interpolate(UtmPoint p1, UtmPoint p2, double ratio) {
        double e = p1.getE() + ratio * (p2.getE() - p1.getE());
        double n = p1.getN() + ratio * (p2.getN() - p1.getN());
        double z = p1.getZ() + ratio * (p2.getZ() - p1.getZ());
        
        return new UtmPoint(e, n, z);
    }
}