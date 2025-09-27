package uo.ml.neural.tracks.core.model;

/**
 * Represents a point in UTM (Universal Transverse Mercator) coordinate system.
 * UTM coordinates provide a more suitable reference for distance calculations
 * and geometric operations on GPS tracks.
 */
public class UtmPoint {
    
    private final double e; // Easting
    private final double n; // Northing
    private final double z; // Altitude
    
    /**
     * Creates a new UTM point with the specified coordinates.
     * 
     * @param e Easting coordinate in meters
     * @param n Northing coordinate in meters
     * @param z Altitude in meters
     */
    public UtmPoint(double e, double n, double z) {
        this.e = e;
        this.n = n;
        this.z = z;
    }
    
    /**
     * Gets the easting coordinate.
     * 
     * @return Easting in meters
     */
    public double getE() {
        return e;
    }
    
    /**
     * Gets the northing coordinate.
     * 
     * @return Northing in meters
     */
    public double getN() {
        return n;
    }
    
    /**
     * Gets the altitude.
     * 
     * @return Altitude in meters
     */
    public double getZ() {
        return z;
    }
    
    @Override
    public String toString() {
        return String.format("UtmPoint{e=%.2f, n=%.2f, z=%.2f}", e, n, z);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof UtmPoint)) return false;
        
        UtmPoint other = (UtmPoint) obj;
        return Double.compare(e, other.e) == 0 &&
               Double.compare(n, other.n) == 0 &&
               Double.compare(z, other.z) == 0;
    }
    
    @Override
    public int hashCode() {
        return java.util.Objects.hash(e, n, z);
    }
}