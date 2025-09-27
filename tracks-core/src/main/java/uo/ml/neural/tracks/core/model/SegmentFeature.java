package uo.ml.neural.tracks.core.model;

/**
 * Represents geometric features extracted from a track segment.
 * These features are used for machine learning analysis of GPS tracks.
 */
public class SegmentFeature {
    
    private final double dh; // 2D horizontal distance
    private final double dz; // Vertical distance (elevation change)
    private final double slope; // Slope calculated as dz/(dh+1e-6)
    
    /**
     * Creates a new segment feature with the specified values.
     * 
     * @param dh 2D horizontal distance in meters
     * @param dz Vertical distance (elevation change) in meters
     * @param slope Slope calculated as dz/(dh+1e-6)
     */
    public SegmentFeature(double dh, double dz, double slope) {
        this.dh = dh;
        this.dz = dz;
        this.slope = slope;
    }
    
    /**
     * Gets the 2D horizontal distance.
     * 
     * @return Horizontal distance in meters
     */
    public double getDh() {
        return dh;
    }
    
    /**
     * Gets the vertical distance (elevation change).
     * 
     * @return Vertical distance in meters
     */
    public double getDz() {
        return dz;
    }
    
    /**
     * Gets the slope.
     * 
     * @return Slope calculated as dz/(dh+1e-6)
     */
    public double getSlope() {
        return slope;
    }
    
    @Override
    public String toString() {
        return String.format("SegmentFeature{dh=%.3f, dz=%.3f, slope=%.6f}", dh, dz, slope);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof SegmentFeature)) return false;
        
        SegmentFeature other = (SegmentFeature) obj;
        return Double.compare(dh, other.dh) == 0 &&
               Double.compare(dz, other.dz) == 0 &&
               Double.compare(slope, other.slope) == 0;
    }
    
    @Override
    public int hashCode() {
        return java.util.Objects.hash(dh, dz, slope);
    }
}