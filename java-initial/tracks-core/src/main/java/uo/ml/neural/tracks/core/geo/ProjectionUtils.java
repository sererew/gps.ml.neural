package uo.ml.neural.tracks.core.geo;

import org.locationtech.proj4j.*;
import uo.ml.neural.tracks.core.model.GpxPoint;
import uo.ml.neural.tracks.core.model.UtmPoint;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for coordinate system transformations between GPS (WGS84) and UTM.
 * Handles UTM zone detection and coordinate conversion for GPS track processing.
 */
public class ProjectionUtils {
    
    private static final CRSFactory CRS_FACTORY = new CRSFactory();
    private static final CoordinateTransformFactory CT_FACTORY = new CoordinateTransformFactory();
    private static final CoordinateReferenceSystem WGS84 = CRS_FACTORY.createFromName("EPSG:4326");
    
    /**
     * Represents a UTM zone with its zone number and hemisphere.
     */
    public static class UtmZone {
        private final int zoneNumber;
        private final boolean isNorthern;
        
        public UtmZone(int zoneNumber, boolean isNorthern) {
            this.zoneNumber = zoneNumber;
            this.isNorthern = isNorthern;
        }
        
        public int getZoneNumber() {
            return zoneNumber;
        }
        
        public boolean isNorthern() {
            return isNorthern;
        }
        
        @Override
        public String toString() {
            return String.format("UTM Zone %d%s", zoneNumber, isNorthern ? "N" : "S");
        }
    }
    
    /**
     * Detects the appropriate UTM zone based on the first GPS point in the list.
     * 
     * @param points List of GPS points (must not be empty)
     * @return UTM zone determined from the first point's longitude and latitude
     * @throws IllegalArgumentException if the points list is empty
     */
    public static UtmZone detectZone(List<GpxPoint> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Points list cannot be empty");
        }
        
        GpxPoint firstPoint = points.get(0);
        int zoneNumber = (int) Math.floor((firstPoint.getLon() + 180) / 6) + 1;
        boolean isNorthern = firstPoint.getLat() >= 0;
        
        return new UtmZone(zoneNumber, isNorthern);
    }
    
    /**
     * Converts a list of GPS points to UTM coordinates using the specified UTM zone.
     * 
     * @param points List of GPS points to convert
     * @param utmZone UTM zone to use for the conversion
     * @return List of UTM points in the specified zone
     * @throws RuntimeException if coordinate transformation fails
     */
    public static List<UtmPoint> toUtm(List<GpxPoint> points, UtmZone utmZone) {
        List<UtmPoint> utmPoints = new ArrayList<>();
        
        try {
            // Create UTM CRS for the specified zone
            String utmCode = String.format("EPSG:326%02d", utmZone.getZoneNumber());
            if (!utmZone.isNorthern()) {
                utmCode = String.format("EPSG:327%02d", utmZone.getZoneNumber());
            }
            
            CoordinateReferenceSystem utmCrs = CRS_FACTORY.createFromName(utmCode);
            CoordinateTransform transform = CT_FACTORY.createTransform(WGS84, utmCrs);
            
            // Transform each point
            for (GpxPoint gpxPoint : points) {
                ProjCoordinate srcCoord = new ProjCoordinate(gpxPoint.getLon(), gpxPoint.getLat());
                ProjCoordinate dstCoord = new ProjCoordinate();
                
                transform.transform(srcCoord, dstCoord);
                
                utmPoints.add(new UtmPoint(dstCoord.x, dstCoord.y, gpxPoint.getAlt()));
            }
            
            return utmPoints;
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to transform coordinates to UTM", e);
        }
    }
}