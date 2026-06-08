package uo.ml.neural.tracks.core.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import uo.ml.neural.tracks.core.model.GpxPoint;

/**
 * Unit tests for GpxUtils class.
 */
class GpxUtilsTest {
    
    @TempDir
    Path tempDir;
    
    @Test
    void testReadGpx() throws IOException {
        // Create a simple GPX file content
        String gpxContent = """
            <?xml version="1.0" encoding="UTF-8"?>
            <gpx version="1.1" creator="test">
              <trk>
                <name>Test Track</name>
                <trkseg>
                  <trkpt lat="43.3614" lon="-5.8593">
                    <ele>100.0</ele>
                  </trkpt>
                  <trkpt lat="43.3615" lon="-5.8594">
                    <ele>105.5</ele>
                  </trkpt>
                  <trkpt lat="43.3616" lon="-5.8595">
                  </trkpt>
                </trkseg>
              </trk>
            </gpx>
            """;
        
        // Write GPX content to temporary file
        Path gpxFile = tempDir.resolve("test.gpx");
        Files.writeString(gpxFile, gpxContent);
        
        // Read GPX file
        List<GpxPoint> points = GpxUtils.readGpx(gpxFile);
        
        // Verify results
        assertEquals(3, points.size());
        
        GpxPoint point1 = points.get(0);
        assertEquals(43.3614, point1.getLat(), 1e-6);
        assertEquals(-5.8593, point1.getLon(), 1e-6);
        assertEquals(100.0, point1.getAlt(), 1e-6);
        
        GpxPoint point2 = points.get(1);
        assertEquals(43.3615, point2.getLat(), 1e-6);
        assertEquals(-5.8594, point2.getLon(), 1e-6);
        assertEquals(105.5, point2.getAlt(), 1e-6);
        
        GpxPoint point3 = points.get(2);
        assertEquals(43.3616, point3.getLat(), 1e-6);
        assertEquals(-5.8595, point3.getLon(), 1e-6);
        assertEquals(0.0, point3.getAlt(), 1e-6); // No elevation specified
    }
    
    @Test
    void testReadGpxNonExistentFile() {
        Path nonExistentFile = tempDir.resolve("nonexistent.gpx");
        
        assertThrows(UncheckedIOException.class, () -> {
            GpxUtils.readGpx(nonExistentFile);
        });
    }
}