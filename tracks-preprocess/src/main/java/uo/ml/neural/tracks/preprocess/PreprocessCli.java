package uo.ml.neural.tracks.preprocess;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import uo.ml.neural.tracks.core.geo.ProjectionUtils;
import uo.ml.neural.tracks.core.io.GpxUtils;
import uo.ml.neural.tracks.core.model.GpxPoint;
import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.core.model.UtmPoint;
import uo.ml.neural.tracks.core.preprocess.FeatureExtractor;
import uo.ml.neural.tracks.core.preprocess.Filters;
import uo.ml.neural.tracks.core.preprocess.Resampler3D;
import uo.ml.neural.tracks.core.preprocess.ZScoreScaler;

/**
 * CLI tool for preprocessing GPS tracks and generating training datasets.
 * Processes folders of GPX files and generates features, labels, and normalization parameters.
 */
@Command(
    name = "tracks-preprocess",
    mixinStandardHelpOptions = true,
    version = "1.0.0-SNAPSHOT",
    description = "Preprocesses GPS tracks from GPX files and generates ML training datasets"
)
public class PreprocessCli implements Callable<Integer> {
    
    @Option(names = {"--input"}, required = true, 
            description = "Root directory containing family subdirectories with GPX files")
    private Path inputDir;
    
    @Option(names = {"--output"}, required = true,
            description = "Output directory for processed datasets")
    private Path outputDir;
    
    @Option(names = {"--step"}, defaultValue = "1.0",
            description = "Resampling step size in meters (default: ${DEFAULT-VALUE})")
    private double stepMeters;
    
    @Option(names = {"--filter"}, defaultValue = "none",
            description = "Altitude filter: median, sgolay, or none (default: ${DEFAULT-VALUE})")
    private FilterType filter;
    
    public enum FilterType {
        median, sgolay, none
    }
    
    public static void main(String[] args) {
        int exitCode = new CommandLine(new PreprocessCli()).execute(args);
        System.exit(exitCode);
    }
    
    @Override
    public Integer call() throws Exception {
        System.out.println("GPS Tracks Preprocessing Tool");
        System.out.println("=============================");
        System.out.printf("Input directory: %s%n", inputDir);
        System.out.printf("Output directory: %s%n", outputDir);
        System.out.printf("Step size: %.1f meters%n", stepMeters);
        System.out.printf("Altitude filter: %s%n", filter);
        System.out.println();
        
        // Validate input directory
        if (!Files.exists(inputDir) || !Files.isDirectory(inputDir)) {
            System.err.println("Error: Input directory does not exist or is not a directory: " + inputDir);
            return 1;
        }
        
        // Create output directory structure
        createOutputDirectories();
        
        // Process all families
        List<Path> familyDirs = findFamilyDirectories();
        if (familyDirs.isEmpty()) {
            System.err.println("Error: No family directories found in input directory");
            return 1;
        }
        
        System.out.printf("Found %d families to process%n", familyDirs.size());
        
        List<SegmentFeature> allFeatures = new ArrayList<>();
        
        for (Path familyDir : familyDirs) {
            String familyName = familyDir.getFileName().toString();
            System.out.printf("Processing family: %s%n", familyName);
            
            try {
                ProcessedFamily processedFamily = processFamily(familyDir, familyName);
                allFeatures.addAll(processedFamily.allNoisyFeatures);
                System.out.printf("  Processed %d tracks, pattern track has %d steps%n", 
                    processedFamily.trackCount, processedFamily.patternSteps);
            } catch (Exception e) {
                System.err.printf("Error processing family %s: %s%n", familyName, e.getMessage());
                e.printStackTrace();
                return 1;
            }
        }
        
        // Compute and save global Z-score scaler
        if (!allFeatures.isEmpty()) {
            ZScoreScaler scaler = ZScoreScaler.fit(allFeatures);
            Path scalerPath = outputDir.resolve("mu_sigma.json");
            scaler.save(scalerPath);
            System.out.printf("Saved global Z-score parameters to: %s%n", scalerPath);
            System.out.printf("Total features for normalization: %d%n", allFeatures.size());
        }
        
        System.out.println("Preprocessing completed successfully!");
        return 0;
    }
    
    private void createOutputDirectories() throws IOException {
        Files.createDirectories(outputDir);
        Files.createDirectories(outputDir.resolve("features"));
        Files.createDirectories(outputDir.resolve("labels"));
        Files.createDirectories(outputDir.resolve("lengths"));
    }
    
    private List<Path> findFamilyDirectories() throws IOException {
        try (var stream = Files.list(inputDir)) {
            return stream
                .filter(Files::isDirectory)
                .collect(Collectors.toList());
        }
    }
    
    private ProcessedFamily processFamily(Path familyDir, String familyName) throws IOException {
        // Find all GPX files in the family directory
        List<Path> gpxFiles;
        try (var stream = Files.list(familyDir)) {
            gpxFiles = stream
                .filter(p -> p.getFileName().toString().toLowerCase().endsWith(".gpx"))
                .collect(Collectors.toList());
        }
        
        if (gpxFiles.isEmpty()) {
            throw new IOException("No GPX files found in family directory: " + familyDir);
        }
        
        // Find pattern file
        Path patternFile = gpxFiles.stream()
            .filter(p -> p.getFileName().toString().contains("_pattern.gpx"))
            .findFirst()
            .orElse(null);
        
        if (patternFile == null) {
            throw new IOException("No pattern file (*_pattern.gpx) found in family directory: " + familyDir);
        }
        
        // Separate noisy tracks from pattern
        List<Path> noisyFiles = gpxFiles.stream()
            .filter(p -> !p.equals(patternFile))
            .collect(Collectors.toList());
        
        System.out.printf("  Found %d noisy tracks and 1 pattern track%n", noisyFiles.size());
        
        // Process pattern track to get labels
        TrackData patternTrack = processTrack(patternFile);
        double[] labels = computeLabels(patternTrack.features);
        
        // Create family output directories
        Path familyFeaturesDir = outputDir.resolve("features").resolve(familyName);
        Path familyLengthsDir = outputDir.resolve("lengths").resolve(familyName);
        Files.createDirectories(familyFeaturesDir);
        Files.createDirectories(familyLengthsDir);
        
        // Save labels
        saveLabels(familyName, labels);
        
        // Process all tracks (including pattern for features)
        List<SegmentFeature> allNoisyFeatures = new ArrayList<>();
        int trackCount = 0;
        
        // Process pattern track
        String patternBaseName = getBaseName(patternFile);
        saveFeaturesCSV(familyFeaturesDir.resolve(patternBaseName + ".csv"), patternTrack.features);
        saveLength(familyLengthsDir.resolve(patternBaseName + ".txt"), patternTrack.features.size());
        trackCount++;
        
        // Process noisy tracks
        for (Path noisyFile : noisyFiles) {
            TrackData trackData = processTrack(noisyFile);
            String baseName = getBaseName(noisyFile);
            
            saveFeaturesCSV(familyFeaturesDir.resolve(baseName + ".csv"), trackData.features);
            saveLength(familyLengthsDir.resolve(baseName + ".txt"), trackData.features.size());
            
            allNoisyFeatures.addAll(trackData.features);
            trackCount++;
        }
        
        return new ProcessedFamily(trackCount, patternTrack.features.size(), allNoisyFeatures);
    }
    
    private TrackData processTrack(Path gpxFile) throws IOException {
        // Read GPX points
        List<GpxPoint> gpxPoints = GpxUtils.readGpx(gpxFile);
        if (gpxPoints.size() < 2) {
            throw new IOException("Track has fewer than 2 points: " + gpxFile);
        }
        
        // Convert to UTM
        ProjectionUtils.UtmZone utmZone = ProjectionUtils.detectZone(gpxPoints);
        List<UtmPoint> utmPoints = ProjectionUtils.toUtm(gpxPoints, utmZone);
        
        // Apply altitude filter if specified
        if (filter != FilterType.none) {
            utmPoints = applyAltitudeFilter(utmPoints);
        }
        
        // Resample by 3D arc length
        List<UtmPoint> resampledPoints = Resampler3D.resampleByArcLength3D(utmPoints, stepMeters);
        
        // Extract features
        List<SegmentFeature> features = FeatureExtractor.computeFeatures(resampledPoints);
        
        return new TrackData(features);
    }
    
    private List<UtmPoint> applyAltitudeFilter(List<UtmPoint> points) {
        // Extract altitude values
        List<Double> altitudes = points.stream()
            .map(UtmPoint::getZ)
            .collect(Collectors.toList());
        
        // Apply filter
        List<Double> filteredAltitudes;
        switch (filter) {
            case median:
                filteredAltitudes = Filters.medianFilter(altitudes, 5);
                break;
            case sgolay:
                filteredAltitudes = Filters.savitzkyGolay(altitudes, 5, 2);
                break;
            default:
                return points; // No filtering
        }
        
        // Create new points with filtered altitudes
        List<UtmPoint> filtered = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            UtmPoint original = points.get(i);
            filtered.add(new UtmPoint(original.getE(), original.getN(), filteredAltitudes.get(i)));
        }
        
        return filtered;
    }
    
    private double[] computeLabels(List<SegmentFeature> features) {
        double distTotal = features.stream().mapToDouble(SegmentFeature::getDh).sum();
        double desnPos = features.stream().mapToDouble(SegmentFeature::getDz).filter(dz -> dz > 0).sum();
        double desnNeg = Math.abs(features.stream().mapToDouble(SegmentFeature::getDz).filter(dz -> dz < 0).sum());
        
        return new double[]{distTotal, desnPos, desnNeg};
    }
    
    private void saveFeaturesCSV(Path csvFile, List<SegmentFeature> features) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("dh,dz,slope"); // Header
        
        for (SegmentFeature feature : features) {
            lines.add(String.format("%.6f,%.6f,%.6f", 
                feature.getDh(), feature.getDz(), feature.getSlope()));
        }
        
        Files.write(csvFile, lines);
    }
    
    private void saveLabels(String familyName, double[] labels) throws IOException {
        Path labelsFile = outputDir.resolve("labels").resolve(familyName + ".csv");
        String content = String.format("dist_total,desn_pos,desn_neg%n%.6f,%.6f,%.6f%n", 
            labels[0], labels[1], labels[2]);
        Files.writeString(labelsFile, content);
    }
    
    private void saveLength(Path lengthFile, int length) throws IOException {
        Files.writeString(lengthFile, String.valueOf(length));
    }
    
    private String getBaseName(Path file) {
        String fileName = file.getFileName().toString();
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
    }
    
    private static class TrackData {
        final List<SegmentFeature> features;
        
        TrackData(List<SegmentFeature> features) {
            this.features = features;
        }
    }
    
    private static class ProcessedFamily {
        final int trackCount;
        final int patternSteps;
        final List<SegmentFeature> allNoisyFeatures;
        
        ProcessedFamily(int trackCount, int patternSteps, List<SegmentFeature> allNoisyFeatures) {
            this.trackCount = trackCount;
            this.patternSteps = patternSteps;
            this.allNoisyFeatures = allNoisyFeatures;
        }
    }
}