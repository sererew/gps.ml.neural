package uo.ml.neural.tracks.core.preprocess;

import org.junit.jupiter.api.Test;
import uo.ml.neural.tracks.core.model.UtmPoint;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Resampler3D class.
 */
class Resampler3DTest {
    
    @Test
    void testResampleByArcLength3D() {
        // Create a simple polyline: straight line from (0,0,0) to (10,0,0)
        List<UtmPoint> originalPoints = Arrays.asList(
            new UtmPoint(0, 0, 0),
            new UtmPoint(5, 0, 0),
            new UtmPoint(10, 0, 0)
        );
        
        // Resample with 2.5m steps
        List<UtmPoint> resampled = Resampler3D.resampleByArcLength3D(originalPoints, 2.5);
        
        // Should have 5 points: at 0, 2.5, 5.0, 7.5, 10.0
        assertEquals(5, resampled.size());
        
        assertEquals(0.0, resampled.get(0).getE(), 1e-6);
        assertEquals(2.5, resampled.get(1).getE(), 1e-6);
        assertEquals(5.0, resampled.get(2).getE(), 1e-6);
        assertEquals(7.5, resampled.get(3).getE(), 1e-6);
        
        // All points should have same N and Z coordinates
        for (UtmPoint point : resampled) {
            assertEquals(0.0, point.getN(), 1e-6);
            assertEquals(0.0, point.getZ(), 1e-6);
        }
    }
    
    @Test
    void testResample3DWithElevation() {
        // Create a line with elevation change: (0,0,0) to (0,0,10)
        List<UtmPoint> originalPoints = Arrays.asList(
            new UtmPoint(0, 0, 0),
            new UtmPoint(0, 0, 10)
        );
        
        // Resample with 5m 3D steps
        List<UtmPoint> resampled = Resampler3D.resampleByArcLength3D(originalPoints, 5.0);
        
        // Should have 3 points: at z=0, z=5, z=10
        assertEquals(3, resampled.size());
        
        assertEquals(0.0, resampled.get(0).getZ(), 1e-6);
        assertEquals(5.0, resampled.get(1).getZ(), 1e-6);
        assertEquals(10.0, resampled.get(2).getZ(), 1e-6);
    }
    
    @Test
    void testResampleSinglePoint() {
        List<UtmPoint> singlePoint = Arrays.asList(new UtmPoint(1, 2, 3));
        
        List<UtmPoint> resampled = Resampler3D.resampleByArcLength3D(singlePoint, 1.0);
        
        assertEquals(1, resampled.size());
        assertEquals(singlePoint.get(0), resampled.get(0));
    }
    
    @Test
    void testResampleEmptyList() {
        assertThrows(IllegalArgumentException.class, () -> {
            Resampler3D.resampleByArcLength3D(Arrays.asList(), 1.0);
        });
    }
    
    @Test
    void testResampleInvalidStepSize() {
        List<UtmPoint> points = Arrays.asList(
            new UtmPoint(0, 0, 0),
            new UtmPoint(1, 0, 0)
        );
        
        assertThrows(IllegalArgumentException.class, () -> {
            Resampler3D.resampleByArcLength3D(points, 0.0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            Resampler3D.resampleByArcLength3D(points, -1.0);
        });
    }
}