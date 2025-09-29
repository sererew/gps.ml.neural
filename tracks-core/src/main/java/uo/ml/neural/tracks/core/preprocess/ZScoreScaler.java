package uo.ml.neural.tracks.core.preprocess;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

import uo.ml.neural.tracks.core.exception.IO;
import uo.ml.neural.tracks.core.model.SegmentFeature;

/**
 * Z-Score scaler for normalizing segment features.
 * Computes mean and standard deviation for each feature dimension and applies 
 * z-score normalization.
 * Supports saving/loading scaler parameters to/from JSON files.
 */
public class ZScoreScaler {
    
    private final double muDh;
    private final double muDz;
    private final double muSlope;
    private final double sigmaDh;
    private final double sigmaDz;
    private final double sigmaSlope;
    
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    
    /**
     * Creates a Z-Score scaler with pre-computed statistics.
     * 
     * @param muDh Mean of dh values
     * @param muDz Mean of dz values
     * @param muSlope Mean of slope values
     * @param sigmaDh Standard deviation of dh values
     * @param sigmaDz Standard deviation of dz values
     * @param sigmaSlope Standard deviation of slope values
     */
    @JsonCreator
    public ZScoreScaler(@JsonProperty("muDh") double muDh,
                        @JsonProperty("muDz") double muDz,
                        @JsonProperty("muSlope") double muSlope,
                        @JsonProperty("sigmaDh") double sigmaDh,
                        @JsonProperty("sigmaDz") double sigmaDz,
                        @JsonProperty("sigmaSlope") double sigmaSlope) {
        this.muDh = muDh;
        this.muDz = muDz;
        this.muSlope = muSlope;
        this.sigmaDh = sigmaDh;
        this.sigmaDz = sigmaDz;
        this.sigmaSlope = sigmaSlope;
    }
    
    /**
     * Fits the scaler by computing mean and standard deviation from training data.
     * 
     * @param features List of segment features to compute statistics from
     * @return New ZScoreScaler fitted to the data
     * @throws IllegalArgumentException if features list is empty
     */
    public static ZScoreScaler fit(List<SegmentFeature> features) {
        if (features.isEmpty()) {
            throw new IllegalArgumentException("Features list cannot be empty");
        }
        
        // Compute means
        double sumDh = 0.0, sumDz = 0.0, sumSlope = 0.0;
        for (SegmentFeature feature : features) {
            sumDh += feature.getDh();
            sumDz += feature.getDz();
            sumSlope += feature.getSlope();
        }
        
        int n = features.size();
        double muDh = sumDh / n;
        double muDz = sumDz / n;
        double muSlope = sumSlope / n;
        
        // Compute standard deviations
        double sumSqDh = 0.0, sumSqDz = 0.0, sumSqSlope = 0.0;
        for (SegmentFeature feature : features) {
            double diffDh = feature.getDh() - muDh;
            double diffDz = feature.getDz() - muDz;
            double diffSlope = feature.getSlope() - muSlope;
            
            sumSqDh += diffDh * diffDh;
            sumSqDz += diffDz * diffDz;
            sumSqSlope += diffSlope * diffSlope;
        }
        
        double sigmaDh = Math.sqrt(sumSqDh / n);
        double sigmaDz = Math.sqrt(sumSqDz / n);
        double sigmaSlope = Math.sqrt(sumSqSlope / n);
        
        // Avoid division by zero
        sigmaDh = Math.max(sigmaDh, 1e-8);
        sigmaDz = Math.max(sigmaDz, 1e-8);
        sigmaSlope = Math.max(sigmaSlope, 1e-8);
        
        return new ZScoreScaler(muDh, muDz, muSlope, sigmaDh, sigmaDz, sigmaSlope);
    }
    
    /**
     * Transforms a segment feature using z-score normalization.
     * 
     * @param feature Input segment feature
     * @return Normalized segment feature
     */
    public SegmentFeature transform(SegmentFeature feature) {
        double normalizedDh = (feature.getDh() - muDh) / sigmaDh;
        double normalizedDz = (feature.getDz() - muDz) / sigmaDz;
        double normalizedSlope = (feature.getSlope() - muSlope) / sigmaSlope;
        
        return new SegmentFeature(normalizedDh, normalizedDz, normalizedSlope);
    }
    
    /**
     * Saves the scaler parameters to a JSON file.
     * 
     * @param path Path to save the JSON file
     */
    public void save(Path path) {
    	IO.shallow(() -> OBJECT_MAPPER.writeValue(path.toFile(), this));
    }
    
    /**
     * Loads a scaler from a JSON file.
     * 
     * @param path Path to the JSON file
     * @return Loaded ZScoreScaler
     * @throws IOException if file cannot be read or parsed
     */
    public static ZScoreScaler load(Path path) {
    	return IO.get(() -> OBJECT_MAPPER.readValue(path.toFile(), ZScoreScaler.class));
    }
    
    // Getters for JSON serialization
    @JsonProperty("muDh")
    public double getMuDh() { return muDh; }
    
    @JsonProperty("muDz")
    public double getMuDz() { return muDz; }
    
    @JsonProperty("muSlope")
    public double getMuSlope() { return muSlope; }
    
    @JsonProperty("sigmaDh")
    public double getSigmaDh() { return sigmaDh; }
    
    @JsonProperty("sigmaDz")
    public double getSigmaDz() { return sigmaDz; }
    
    @JsonProperty("sigmaSlope")
    public double getSigmaSlope() { return sigmaSlope; }
}