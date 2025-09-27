package uo.ml.neural.tracks.core.preprocess;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import uo.ml.neural.tracks.core.model.SegmentFeature;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ZScoreScaler class.
 */
class ZScoreScalerTest {
    
    @TempDir
    Path tempDir;
    
    @Test
    void testFitAndTransform() {
        // Create test features with known statistics
        List<SegmentFeature> features = Arrays.asList(
            new SegmentFeature(1.0, 2.0, 0.5),
            new SegmentFeature(3.0, 4.0, 1.5),
            new SegmentFeature(5.0, 6.0, 2.5)
        );
        
        // Fit scaler
        ZScoreScaler scaler = ZScoreScaler.fit(features);
        
        // Check computed means (should be 3.0, 4.0, 1.5)
        assertEquals(3.0, scaler.getMuDh(), 1e-6);
        assertEquals(4.0, scaler.getMuDz(), 1e-6);
        assertEquals(1.5, scaler.getMuSlope(), 1e-6);
        
        // Check computed standard deviations
        // For dh: sqrt(((1-3)^2 + (3-3)^2 + (5-3)^2) / 3) = sqrt(8/3) ≈ 1.633
        assertEquals(Math.sqrt(8.0/3.0), scaler.getSigmaDh(), 1e-6);
        assertEquals(Math.sqrt(8.0/3.0), scaler.getSigmaDz(), 1e-6);
        assertEquals(Math.sqrt(2.0/3.0), scaler.getSigmaSlope(), 1e-6);
        
        // Transform a feature
        SegmentFeature testFeature = new SegmentFeature(3.0, 4.0, 1.5);
        SegmentFeature normalized = scaler.transform(testFeature);
        
        // Mean values should become 0
        assertEquals(0.0, normalized.getDh(), 1e-6);
        assertEquals(0.0, normalized.getDz(), 1e-6);
        assertEquals(0.0, normalized.getSlope(), 1e-6);
    }
    
    @Test
    void testSaveAndLoad() throws IOException {
        // Create and fit a scaler
        List<SegmentFeature> features = Arrays.asList(
            new SegmentFeature(10.0, 5.0, 0.1),
            new SegmentFeature(20.0, 10.0, 0.2),
            new SegmentFeature(30.0, 15.0, 0.3)
        );
        
        ZScoreScaler originalScaler = ZScoreScaler.fit(features);
        
        // Save to file
        Path scalerFile = tempDir.resolve("scaler.json");
        originalScaler.save(scalerFile);
        
        // Load from file
        ZScoreScaler loadedScaler = ZScoreScaler.load(scalerFile);
        
        // Verify all parameters are preserved
        assertEquals(originalScaler.getMuDh(), loadedScaler.getMuDh(), 1e-6);
        assertEquals(originalScaler.getMuDz(), loadedScaler.getMuDz(), 1e-6);
        assertEquals(originalScaler.getMuSlope(), loadedScaler.getMuSlope(), 1e-6);
        assertEquals(originalScaler.getSigmaDh(), loadedScaler.getSigmaDh(), 1e-6);
        assertEquals(originalScaler.getSigmaDz(), loadedScaler.getSigmaDz(), 1e-6);
        assertEquals(originalScaler.getSigmaSlope(), loadedScaler.getSigmaSlope(), 1e-6);
        
        // Verify both scalers produce same transformation
        SegmentFeature testFeature = new SegmentFeature(25.0, 12.5, 0.25);
        SegmentFeature normalized1 = originalScaler.transform(testFeature);
        SegmentFeature normalized2 = loadedScaler.transform(testFeature);
        
        assertEquals(normalized1.getDh(), normalized2.getDh(), 1e-6);
        assertEquals(normalized1.getDz(), normalized2.getDz(), 1e-6);
        assertEquals(normalized1.getSlope(), normalized2.getSlope(), 1e-6);
    }
    
    @Test
    void testFitEmptyList() {
        assertThrows(IllegalArgumentException.class, () -> {
            ZScoreScaler.fit(Arrays.asList());
        });
    }
    
    @Test
    void testLoadNonExistentFile() {
        Path nonExistentFile = tempDir.resolve("nonexistent.json");
        
        assertThrows(IOException.class, () -> {
            ZScoreScaler.load(nonExistentFile);
        });
    }
}