package uo.ml.neural.tracks.core.io;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import io.jenetics.jpx.GPX;
import io.jenetics.jpx.Track;
import io.jenetics.jpx.TrackSegment;
import io.jenetics.jpx.WayPoint;
import uo.ml.neural.tracks.core.exception.IO;
import uo.ml.neural.tracks.core.model.GpxPoint;

/**
 * Utility class for reading and processing GPX files.
 * Uses the JPX library to parse GPX data and convert it to internal GpxPoint format.
 */
public class GpxUtils {
    
    /**
     * Reads a GPX file and extracts all track points as a list of GpxPoint objects.
     * 
     * @param file Path to the GPX file
     * @return List of GPS points extracted from all tracks and segments in the file
     */
    public static List<GpxPoint> readGpx(Path file) {
        List<GpxPoint> points = new ArrayList<>();
        
        GPX gpx = IO.get(() -> GPX.read(file) );
        
        // Extract points from all tracks and segments
        for (Track track : gpx.getTracks()) {
            for (TrackSegment segment : track.getSegments()) {
                for (WayPoint wayPoint : segment.getPoints()) {
                    double lat = wayPoint.getLatitude().doubleValue();
                    double lon = wayPoint.getLongitude().doubleValue();
                    double alt = wayPoint.getElevation()
                        .map(elevation -> elevation.doubleValue())
                        .orElse(0.0);
                    
                    points.add(new GpxPoint(lat, lon, alt));
                }
            }
        }
        
        return points;
    }
}