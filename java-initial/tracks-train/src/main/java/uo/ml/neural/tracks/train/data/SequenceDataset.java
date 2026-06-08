package uo.ml.neural.tracks.train.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.core.exception.IO;
import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.core.preprocess.Padding;
import uo.ml.neural.tracks.core.preprocess.ZScoreScaler;

/**
 * Dataset loader for sequence data from preprocessed GPS tracks.
 * Loads features, labels, and lengths from the preprocessing output structure.
 */
public class SequenceDataset {
    
    private final INDArray dataMatrix3D;		// Shape: [batchSize, nFeatures, maxLength]
    private final INDArray dataMask2D;			// Shape: [batchSize, maxLength]
    private final INDArray expectedValues2D;	// Shape: [batchSize, nLabels]
    private final List<String> trackNames;
    private final List<String> familyNames;
    
    private SequenceDataset(
    		INDArray data, 
    		INDArray dataMask, 
    		INDArray expectedValues, 
            List<String> trackNames, 
            List<String> familyNames) {
    	
        this.dataMatrix3D = data;
        this.dataMask2D = dataMask;
        this.expectedValues2D = expectedValues;
        this.trackNames = new ArrayList<>(trackNames);
        this.familyNames = new ArrayList<>(familyNames);
    }
    
    // Getters
    public INDArray getDataMatrix3D() { return dataMatrix3D; }
    public INDArray getDataMask2D() { return dataMask2D; }
    public INDArray getExpectecValues2D() { return expectedValues2D; }
    public int getNumTracks() { return (int) dataMatrix3D.size(0); }
    public int getNumFeatures() { return (int) dataMatrix3D.size(1); }
    public int getNumPointsPerTrack() { return (int) dataMatrix3D.size(2); }
    public List<String> getTrackNames() { return trackNames; }
    public List<String> getFamilyNames() { return familyNames; }
    
    /**
     * Loads dataset from preprocessed directory structure.
     * 
     * @param processedDir Directory containing features/, labels/ and mu_sigma.json
     * @return Loaded and normalized dataset
     * @throws IOException if data cannot be loaded
     */
    public static SequenceDataset load(Path processedDir) {
        return load(processedDir, null);
    }
    
    /**
     * Loads dataset from preprocessed directory structure, excluding specified families.
     * 
     * @param dataDir Directory containing features/, labels/ and mu_sigma.json
     * @param families Families to load for LOFO
     * @return Loaded and normalized dataset
     * @throws IOException if data cannot be loaded
     */
    public static SequenceDataset load(Path dataDir, List<String> families) {
        Path featuresDir = dataDir.resolve("features");
        Path labelsDir = dataDir.resolve("labels");
        Path scalerPath = dataDir.resolve("mu_sigma.json");
        
        if (!Files.exists(featuresDir) 
        		|| !Files.exists(labelsDir) 
        		|| !Files.exists(scalerPath)) {
            throw new CommandException("Missing required directories "
            		+ "or files in: " + dataDir);
        }
        
        // Load Z-score scaler
        ZScoreScaler scaler = ZScoreScaler.load(scalerPath);
        
        // Find all families
        List<String> allFamilies = findAllFamilies(featuresDir);
        
        // Filter out excluded families
        List<String> familiesToProcess;
        if (families == null || families.isEmpty()) {
			familiesToProcess = allFamilies;
		} else {
			familiesToProcess = allFamilies.stream()
	            .filter(family -> families.contains(family))
	            .toList();
		}
        if (familiesToProcess.isEmpty()) {
            throw new CommandException("No families to process after exclusions");
        }
        
        // Load all data
        List<TrackData> allTracks = new ArrayList<>();
        Map<String, float[]> familyLabels = new HashMap<>();
        for (String family : familiesToProcess) {
            // Load family labels
            float[] labels = loadFamilyLabels(labelsDir.resolve(family + ".csv"));
            familyLabels.put(family, labels);
            
            // Load tracks for this family
            Path familyFeaturesDir = featuresDir.resolve(family);
            List<Path> csvFiles = IO.get(() -> Files.list(familyFeaturesDir))
            		.filter(p -> p.getFileName().toString().endsWith(".csv"))
            		.toList();
            
            for (Path csvFile : csvFiles) {
                String trackName = getBaseName(csvFile);
                List<SegmentFeature> rawFeatures = loadTrackFeatures(csvFile);
                
                // Apply normalization
                List<SegmentFeature> normalizedFeatures = rawFeatures.stream()
                    .map(scaler::transform)
                    .toList();
                
                allTracks.add(
                		new TrackData(family + "/" + trackName, 
                				family, 
                                normalizedFeatures, 
                                labels
                             )
                	);
            }
        }
        
        if (allTracks.isEmpty()) {
            throw new CommandException("No tracks loaded");
        }
        
        // Convert to INDArrays
        return convertToArrays(allTracks);
    }

	private static List<String> findAllFamilies(Path featuresDir) {
		List<String> allFamilies;
        try (var stream = IO.get(() -> Files.list(featuresDir))) {
            allFamilies = stream
                .filter(Files::isDirectory)
                .map(p -> p.getFileName().toString())
                .toList();
        }
		return allFamilies;
	}
    
    private static float[] loadFamilyLabels(Path labelsFile) {
        List<String> lines = IO.get(() -> Files.readAllLines(labelsFile));
        if (lines.size() < 2) {
            throw new CommandException("Invalid labels file format: " + labelsFile);
        }
        
        String[] values = lines.get(1).split(","); // Skip header
        if (values.length != 3) {
            throw new CommandException("Expected 3 label values, got " 
            		+ values.length + " in: " + labelsFile);
        }
        
        return new float[]{
            Float.parseFloat(values[0]), // dist_total
            Float.parseFloat(values[1]), // desn_pos
            Float.parseFloat(values[2])  // desn_neg
        };
    }
    
    private static List<SegmentFeature> loadTrackFeatures(Path csvFile) {
        List<String> lines = IO.get(() -> Files.readAllLines(csvFile));
        List<SegmentFeature> features = new ArrayList<>();
        
        for (int i = 1; i < lines.size(); i++) { // Skip header
            String[] values = lines.get(i).split(",");
            if (values.length != 3) {
                throw new CommandException("Expected 3 feature values, got " 
                			+ values.length + " at line " + i
                		);
            }
            
            float dh = Float.parseFloat(values[0]);
            float dz = Float.parseFloat(values[1]);
            float slope = Float.parseFloat(values[2]);
            
            features.add(new SegmentFeature(dh, dz, slope));
        }
        
        return features;
    }
    
    private static SequenceDataset convertToArrays(List<TrackData> tracks) {
        int batchSize = tracks.size();
        int nFeatures = 3; // dh, dz, slope
        int maxLength = tracks.stream().mapToInt(t -> t.features.size()).max().orElse(1);
        
        // Create arrays
        INDArray features = Nd4j.zeros(batchSize, nFeatures, maxLength);
        INDArray featuresMask = Nd4j.zeros(batchSize, maxLength);
        INDArray labels = Nd4j.zeros(batchSize, 3);
        
        List<String> trackNames = new ArrayList<>();
        List<String> familyNames = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            TrackData track = tracks.get(i);
            
            // Pad features
            float[][] paddedFeatures = Padding.padFeatures(track.features, maxLength);
            float[] mask = Padding.makeMask(track.features.size(), maxLength);
            
            // Fill features array [batch, nFeatures, time]
            for (int t = 0; t < maxLength; t++) {
                features.putScalar(new int[]{i, 0, t}, paddedFeatures[t][0]); // dh
                features.putScalar(new int[]{i, 1, t}, paddedFeatures[t][1]); // dz
                features.putScalar(new int[]{i, 2, t}, paddedFeatures[t][2]); // slope
                featuresMask.putScalar(new int[]{i, t}, mask[t]);
            }
            
            // Fill labels
            labels.putScalar(new int[]{i, 0}, track.labels[0]);
            labels.putScalar(new int[]{i, 1}, track.labels[1]);
            labels.putScalar(new int[]{i, 2}, track.labels[2]);
            
            trackNames.add(track.trackName);
            familyNames.add(track.familyName);
        }
        
        return new SequenceDataset(features, featuresMask, labels, trackNames, familyNames);
    }
    
    private static String getBaseName(Path file) {
        String fileName = file.getFileName().toString();
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
    }
    
    private static record TrackData (
        		String trackName, 
        		String familyName, 
        		List<SegmentFeature> features, 
                float[] labels
           ) {}
    
}