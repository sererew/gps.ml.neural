package uo.ml.neural.tracks.core.preprocess;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import uo.ml.neural.tracks.core.model.SegmentFeature;

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
            new SegmentFeature(1.0f, 2.0f, 0.5f),
            new SegmentFeature(3.0f, 4.0f, 1.5f),
            new SegmentFeature(5.0f, 6.0f, 2.5f)
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
        SegmentFeature testFeature = new SegmentFeature(3.0f, 4.0f, 1.5f);
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
            new SegmentFeature(10.0f, 5.0f, 0.1f),
            new SegmentFeature(20.0f, 10.0f, 0.2f),
            new SegmentFeature(30.0f, 15.0f, 0.3f)
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
        SegmentFeature testFeature = new SegmentFeature(25.0f, 12.5f, 0.25f);
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
        
        assertThrows(UncheckedIOException.class, () -> {
            ZScoreScaler.load(nonExistentFile);
        });
    }
}