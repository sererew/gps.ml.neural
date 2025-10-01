package uo.ml.neural.tracks.core.preprocess;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

import uo.ml.neural.tracks.core.model.SegmentFeature;

/**
 * Unit tests for Padding class.
 */
class PaddingTest {
    
    @Test
    void testPadFeatures() {
        List<SegmentFeature> features = Arrays.asList(
            new SegmentFeature(1.0f, 2.0f, 0.5f),
            new SegmentFeature(3.0f, 4.0f, 1.5f)
        );
        
        float[][] padded = Padding.padFeatures(features, 4);
        
        // Check dimensions
        assertEquals(4, padded.length);
        assertEquals(3, padded[0].length);
        
        // Check actual feature values
        assertEquals(1.0, padded[0][0], 1e-6); // dh
        assertEquals(2.0, padded[0][1], 1e-6); // dz
        assertEquals(0.5, padded[0][2], 1e-6); // slope
        
        assertEquals(3.0, padded[1][0], 1e-6);
        assertEquals(4.0, padded[1][1], 1e-6);
        assertEquals(1.5, padded[1][2], 1e-6);
        
        // Check padding (should be zeros)
        assertEquals(0.0, padded[2][0], 1e-6);
        assertEquals(0.0, padded[2][1], 1e-6);
        assertEquals(0.0, padded[2][2], 1e-6);
        
        assertEquals(0.0, padded[3][0], 1e-6);
        assertEquals(0.0, padded[3][1], 1e-6);
        assertEquals(0.0, padded[3][2], 1e-6);
    }
    
    @Test
    void testPadFeaturesExactLength() {
        List<SegmentFeature> features = Arrays.asList(
            new SegmentFeature(1.0f, 2.0f, 0.5f),
            new SegmentFeature(3.0f, 4.0f, 1.5f)
        );
        
        float[][] padded = Padding.padFeatures(features, 2);
        
        assertEquals(2, padded.length);
        assertEquals(1.0, padded[0][0], 1e-6);
        assertEquals(3.0, padded[1][0], 1e-6);
    }
    
    @Test
    void testPadFeaturesEmpty() {
        float[][] padded = Padding.padFeatures(Arrays.asList(), 3);
        
        assertEquals(3, padded.length);
        // All should be zeros
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(0.0, padded[i][j], 1e-6);
            }
        }
    }
    
    @Test
    void testMakeMask() {
        float[] mask = Padding.makeMask(3, 5);
        
        assertEquals(5, mask.length);
        
        // First 3 should be 1.0 (real data)
        assertEquals(1.0, mask[0], 1e-6);
        assertEquals(1.0, mask[1], 1e-6);
        assertEquals(1.0, mask[2], 1e-6);
        
        // Last 2 should be 0.0 (padding)
        assertEquals(0.0, mask[3], 1e-6);
        assertEquals(0.0, mask[4], 1e-6);
    }
    
    @Test
    void testMakeMaskFullLength() {
        float[] mask = Padding.makeMask(3, 3);
        
        assertEquals(3, mask.length);
        for (int i = 0; i < 3; i++) {
            assertEquals(1.0, mask[i], 1e-6);
        }
    }
    
    @Test
    void testMakeMaskZeroLength() {
        float[] mask = Padding.makeMask(0, 3);
        
        assertEquals(3, mask.length);
        for (int i = 0; i < 3; i++) {
            assertEquals(0.0, mask[i], 1e-6);
        }
    }
    
    @Test
    void testPadFeaturesInvalidMaxLen() {
        List<SegmentFeature> features = Arrays.asList(
            new SegmentFeature(1.0f, 2.0f, 0.5f)
        );
        
        assertThrows(IllegalArgumentException.class, () -> {
            Padding.padFeatures(features, 0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            Padding.padFeatures(features, -1);
        });
    }
    
    @Test
    void testPadFeaturesSequenceTooLong() {
        List<SegmentFeature> features = Arrays.asList(
            new SegmentFeature(1.0f, 2.0f, 0.5f),
            new SegmentFeature(3.0f, 4.0f, 1.5f),
            new SegmentFeature(5.0f, 6.0f, 2.5f)
        );
        
        assertThrows(IllegalArgumentException.class, () -> {
            Padding.padFeatures(features, 2); // maxLen < sequence length
        });
    }
    
    @Test
    void testMakeMaskInvalidParameters() {
        assertThrows(IllegalArgumentException.class, () -> {
            Padding.makeMask(-1, 5);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            Padding.makeMask(5, -1);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            Padding.makeMask(5, 3); // realLen > maxLen
        });
    }
}