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
    
    private final float muDh;
    private final float muDz;
    private final float muSlope;
    private final float sigmaDh;
    private final float sigmaDz;
    private final float sigmaSlope;
    
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
    public ZScoreScaler(@JsonProperty("muDh") float muDh,
                        @JsonProperty("muDz") float muDz,
                        @JsonProperty("muSlope") float muSlope,
                        @JsonProperty("sigmaDh") float sigmaDh,
                        @JsonProperty("sigmaDz") float sigmaDz,
                        @JsonProperty("sigmaSlope") float sigmaSlope) {
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
        float sumDh = 0.0f;
        float sumDz = 0.0f;
        float sumSlope = 0.0f;
        for (SegmentFeature feature : features) {
            sumDh += feature.getDh();
            sumDz += feature.getDz();
            sumSlope += feature.getSlope();
        }
        
        int n = features.size();
        float muDh = sumDh / n;
        float muDz = sumDz / n;
        float muSlope = sumSlope / n;
        
        // Compute standard deviations
        float sumSqDh = 0.0f;
        float sumSqDz = 0.0f;
        float sumSqSlope = 0.0f;
        for (SegmentFeature feature : features) {
            float diffDh = feature.getDh() - muDh;
            float diffDz = feature.getDz() - muDz;
            float diffSlope = feature.getSlope() - muSlope;
            
            sumSqDh += diffDh * diffDh;
            sumSqDz += diffDz * diffDz;
            sumSqSlope += diffSlope * diffSlope;
        }
        
        float sigmaDh = (float) Math.sqrt(sumSqDh / n);
        float sigmaDz = (float) Math.sqrt(sumSqDz / n);
        float sigmaSlope = (float) Math.sqrt(sumSqSlope / n);
        
        // Avoid division by zero
        sigmaDh = (float) Math.max(sigmaDh, 1e-8);
        sigmaDz = (float) Math.max(sigmaDz, 1e-8);
        sigmaSlope = (float) Math.max(sigmaSlope, 1e-8);
        
        return new ZScoreScaler(muDh, muDz, muSlope, sigmaDh, sigmaDz, sigmaSlope);
    }
    
    /**
     * Transforms a segment feature using z-score normalization.
     * 
     * @param feature Input segment feature
     * @return Normalized segment feature
     */
    public SegmentFeature transform(SegmentFeature feature) {
        float normalizedDh = (feature.getDh() - muDh) / sigmaDh;
        float normalizedDz = (feature.getDz() - muDz) / sigmaDz;
        float normalizedSlope = (feature.getSlope() - muSlope) / sigmaSlope;
        
        return new SegmentFeature(normalizedDh, normalizedDz, normalizedSlope);
    }
    
    /**
     * Saves the scaler parameters to a JSON file.
     * 
     * @param path Path to save the JSON file
     */
    public void save(Path path) {
    	IO.exec(() -> OBJECT_MAPPER.writeValue(path.toFile(), this));
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
    public float getMuDh() { return muDh; }
    
    @JsonProperty("muDz")
    public float getMuDz() { return muDz; }
    
    @JsonProperty("muSlope")
    public float getMuSlope() { return muSlope; }
    
    @JsonProperty("sigmaDh")
    public float getSigmaDh() { return sigmaDh; }
    
    @JsonProperty("sigmaDz")
    public float getSigmaDz() { return sigmaDz; }
    
    @JsonProperty("sigmaSlope")
    public float getSigmaSlope() { return sigmaSlope; }
}