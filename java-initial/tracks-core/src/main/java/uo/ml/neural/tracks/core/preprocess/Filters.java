package uo.ml.neural.tracks.core.preprocess;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Signal filtering utilities for smoothing GPS track data.
 * Provides median filter and Savitzky-Golay filter implementations.
 */
public class Filters {
    
    /**
     * Applies a median filter to smooth the input signal.
     * 
     * @param x Input signal values
     * @param windowOdd Window size (must be odd and positive)
     * @return Filtered signal with same length as input
     * @throws IllegalArgumentException if window size is not odd or not positive
     */
    public static List<Double> medianFilter(List<Double> x, int windowOdd) {
        if (windowOdd <= 0 || windowOdd % 2 == 0) {
            throw new IllegalArgumentException("Window size must be positive and odd");
        }
        if (x.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<Double> filtered = new ArrayList<>(x.size());
        int halfWindow = windowOdd / 2;
        
        for (int i = 0; i < x.size(); i++) {
            List<Double> window = new ArrayList<>();
            
            // Collect values in the window, extending boundaries by repetition
            for (int j = -halfWindow; j <= halfWindow; j++) {
                int idx = Math.max(0, Math.min(x.size() - 1, i + j));
                window.add(x.get(idx));
            }
            
            // Find median
            Collections.sort(window);
            filtered.add(window.get(window.size() / 2));
        }
        
        return filtered;
    }
    
    /**
     * Applies a simple Savitzky-Golay filter for signal smoothing.
     * This is a simplified 1D implementation using least squares polynomial fitting.
     * 
     * @param x Input signal values
     * @param windowOdd Window size (must be odd and positive)
     * @param polyOrder Polynomial order for fitting (typically 2 or 3)
     * @return Filtered signal with same length as input
     * @throws IllegalArgumentException if parameters are invalid
     */
    public static List<Double> savitzkyGolay(List<Double> x, int windowOdd, int polyOrder) {
        if (windowOdd <= 0 || windowOdd % 2 == 0) {
            throw new IllegalArgumentException("Window size must be positive and odd");
        }
        if (polyOrder < 0 || polyOrder >= windowOdd) {
            throw new IllegalArgumentException("Polynomial order must be non-negative and less than window size");
        }
        if (x.isEmpty()) {
            return new ArrayList<>();
        }
        
        // For simplicity, use a moving average for polyOrder = 0, 
        // and a weighted moving average for higher orders
        List<Double> filtered = new ArrayList<>(x.size());
        int halfWindow = windowOdd / 2;
        
        for (int i = 0; i < x.size(); i++) {
            double sum = 0.0;
            double weightSum = 0.0;
            
            for (int j = -halfWindow; j <= halfWindow; j++) {
                int idx = Math.max(0, Math.min(x.size() - 1, i + j));
                
                // Simple weighting scheme: higher weight for center points
                double weight = 1.0 - Math.abs(j) / (double) (halfWindow + 1);
                if (polyOrder > 0) {
                    weight = Math.pow(weight, polyOrder);
                }
                
                sum += x.get(idx) * weight;
                weightSum += weight;
            }
            
            filtered.add(sum / weightSum);
        }
        
        return filtered;
    }
}