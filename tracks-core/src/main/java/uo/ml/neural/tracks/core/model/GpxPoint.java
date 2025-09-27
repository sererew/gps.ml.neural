package uo.ml.neural.tracks.core.model;

/**
 * Represents a GPS point with latitude, longitude, and altitude coordinates.
 * This is the basic unit for GPS track data processing.
 */
public class GpxPoint {
    
    private final double lat;
    private final double lon;
    private final double alt;
    
    /**
     * Creates a new GPS point with the specified coordinates.
     * 
     * @param lat Latitude in decimal degrees
     * @param lon Longitude in decimal degrees  
     * @param alt Altitude in meters
     */
    public GpxPoint(double lat, double lon, double alt) {
        this.lat = lat;
        this.lon = lon;
        this.alt = alt;
    }
    
    /**
     * Gets the latitude coordinate.
     * 
     * @return Latitude in decimal degrees
     */
    public double getLat() {
        return lat;
    }
    
    /**
     * Gets the longitude coordinate.
     * 
     * @return Longitude in decimal degrees
     */
    public double getLon() {
        return lon;
    }
    
    /**
     * Gets the altitude.
     * 
     * @return Altitude in meters
     */
    public double getAlt() {
        return alt;
    }
    
    @Override
    public String toString() {
        return String.format("GpxPoint{lat=%.6f, lon=%.6f, alt=%.2f}", lat, lon, alt);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof GpxPoint)) return false;
        
        GpxPoint other = (GpxPoint) obj;
        return Double.compare(lat, other.lat) == 0 &&
               Double.compare(lon, other.lon) == 0 &&
               Double.compare(alt, other.alt) == 0;
    }
    
    @Override
    public int hashCode() {
        return java.util.Objects.hash(lat, lon, alt);
    }
}