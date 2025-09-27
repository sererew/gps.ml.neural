package uo.ml.neural.tracks.train.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.core.preprocess.Padding;
import uo.ml.neural.tracks.core.preprocess.ZScoreScaler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Dataset loader for sequence data from preprocessed GPS tracks.
 * Loads features, labels, and lengths from the preprocessing output structure.
 */
public class SequenceDataset {
    
    private final INDArray features;
    private final INDArray featuresMask;
    private final INDArray labels;
    private final List<String> trackNames;
    private final List<String> familyNames;
    
    private SequenceDataset(INDArray features, INDArray featuresMask, INDArray labels, 
                           List<String> trackNames, List<String> familyNames) {
        this.features = features;
        this.featuresMask = featuresMask;
        this.labels = labels;
        this.trackNames = new ArrayList<>(trackNames);
        this.familyNames = new ArrayList<>(familyNames);
    }
    
    /**
     * Loads dataset from preprocessed directory structure.
     * 
     * @param processedDir Directory containing features/, labels/, lengths/, and mu_sigma.json
     * @return Loaded and normalized dataset
     * @throws IOException if data cannot be loaded
     */
    public static SequenceDataset load(Path processedDir) throws IOException {
        return load(processedDir, null);
    }
    
    /**
     * Loads dataset from preprocessed directory structure, excluding specified families.
     * 
     * @param processedDir Directory containing features/, labels/, lengths/, and mu_sigma.json
     * @param excludeFamilies Families to exclude from loading (for LOFO), null to include all
     * @return Loaded and normalized dataset
     * @throws IOException if data cannot be loaded
     */
    public static SequenceDataset load(Path processedDir, List<String> excludeFamilies) throws IOException {
        Path featuresDir = processedDir.resolve("features");
        Path labelsDir = processedDir.resolve("labels");
        Path lengthsDir = processedDir.resolve("lengths");
        Path scalerPath = processedDir.resolve("mu_sigma.json");
        
        if (!Files.exists(featuresDir) || !Files.exists(labelsDir) || 
            !Files.exists(lengthsDir) || !Files.exists(scalerPath)) {
            throw new IOException("Missing required directories or files in: " + processedDir);
        }
        
        // Load Z-score scaler
        ZScoreScaler scaler = ZScoreScaler.load(scalerPath);
        
        // Find all families
        List<String> allFamilies;
        try (var stream = Files.list(featuresDir)) {
            allFamilies = stream
                .filter(Files::isDirectory)
                .map(p -> p.getFileName().toString())
                .toList();
        }
        
        // Filter out excluded families
        List<String> familiesToProcess = allFamilies.stream()
            .filter(family -> excludeFamilies == null || !excludeFamilies.contains(family))
            .toList();
        
        if (familiesToProcess.isEmpty()) {
            throw new IOException("No families to process after exclusions");
        }
        
        // Load all data
        List<TrackData> allTracks = new ArrayList<>();
        Map<String, double[]> familyLabels = new HashMap<>();
        
        for (String family : familiesToProcess) {
            // Load family labels
            double[] labels = loadFamilyLabels(labelsDir.resolve(family + ".csv"));
            familyLabels.put(family, labels);
            
            // Load tracks for this family
            Path familyFeaturesDir = featuresDir.resolve(family);
            Path familyLengthsDir = lengthsDir.resolve(family);
            
            try (var stream = Files.list(familyFeaturesDir)) {
                List<Path> csvFiles = stream
                    .filter(p -> p.getFileName().toString().endsWith(".csv"))
                    .toList();
                
                for (Path csvFile : csvFiles) {
                    String trackName = getBaseName(csvFile);
                    Path lengthFile = familyLengthsDir.resolve(trackName + ".txt");
                    
                    List<SegmentFeature> rawFeatures = loadTrackFeatures(csvFile);
                    int length = loadTrackLength(lengthFile);
                    
                    // Apply normalization
                    List<SegmentFeature> normalizedFeatures = rawFeatures.stream()
                        .map(scaler::transform)
                        .toList();
                    
                    allTracks.add(
                    		new TrackData(family + "/" + trackName, 
                    				family, 
                                    normalizedFeatures, 
                                    length, 
                                    labels
                                 )
                    	);
                }
            }
        }
        
        if (allTracks.isEmpty()) {
            throw new IOException("No tracks loaded");
        }
        
        // Convert to INDArrays
        return convertToArrays(allTracks);
    }
    
    private static double[] loadFamilyLabels(Path labelsFile) throws IOException {
        List<String> lines = Files.readAllLines(labelsFile);
        if (lines.size() < 2) {
            throw new IOException("Invalid labels file format: " + labelsFile);
        }
        
        String[] values = lines.get(1).split(","); // Skip header
        if (values.length != 3) {
            throw new IOException("Expected 3 label values, got " + values.length + " in: " + labelsFile);
        }
        
        return new double[]{
            Double.parseDouble(values[0]), // dist_total
            Double.parseDouble(values[1]), // desn_pos
            Double.parseDouble(values[2])  // desn_neg
        };
    }
    
    private static List<SegmentFeature> loadTrackFeatures(Path csvFile) throws IOException {
        List<String> lines = Files.readAllLines(csvFile);
        List<SegmentFeature> features = new ArrayList<>();
        
        for (int i = 1; i < lines.size(); i++) { // Skip header
            String[] values = lines.get(i).split(",");
            if (values.length != 3) {
                throw new IOException("Expected 3 feature values, got " + values.length + " at line " + i);
            }
            
            double dh = Double.parseDouble(values[0]);
            double dz = Double.parseDouble(values[1]);
            double slope = Double.parseDouble(values[2]);
            
            features.add(new SegmentFeature(dh, dz, slope));
        }
        
        return features;
    }
    
    private static int loadTrackLength(Path lengthFile) throws IOException {
        String content = Files.readString(lengthFile).trim();
        return Integer.parseInt(content);
    }
    
    private static SequenceDataset convertToArrays(List<TrackData> tracks) {
        int batchSize = tracks.size();
        int maxLength = tracks.stream().mapToInt(t -> t.features.size()).max().orElse(1);
        int nFeatures = 3; // dh, dz, slope
        
        // Create arrays
        INDArray features = Nd4j.zeros(batchSize, nFeatures, maxLength);
        INDArray featuresMask = Nd4j.zeros(batchSize, maxLength);
        INDArray labels = Nd4j.zeros(batchSize, 3);
        
        List<String> trackNames = new ArrayList<>();
        List<String> familyNames = new ArrayList<>();
        
        for (int i = 0; i < batchSize; i++) {
            TrackData track = tracks.get(i);
            
            // Pad features
            double[][] paddedFeatures = Padding.padFeatures(track.features, maxLength);
            double[] mask = Padding.makeMask(track.features.size(), maxLength);
            
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
    
    // Getters
    public INDArray getFeatures() { return features; }
    public INDArray getFeaturesMask() { return featuresMask; }
    public INDArray getLabels() { return labels; }
    public List<String> getTrackNames() { return trackNames; }
    public List<String> getFamilyNames() { return familyNames; }
    public int getBatchSize() { return (int) features.size(0); }
    public int getMaxSequenceLength() { return (int) features.size(2); }
    public int getNumFeatures() { return (int) features.size(1); }
    
    private static class TrackData {
        final String trackName;
        final String familyName;
        final List<SegmentFeature> features;
        final int length;
        final double[] labels;
        
        TrackData(String trackName, String familyName, List<SegmentFeature> features, 
                  int length, double[] labels) {
            this.trackName = trackName;
            this.familyName = familyName;
            this.features = features;
            this.length = length;
            this.labels = labels;
        }
    }
}