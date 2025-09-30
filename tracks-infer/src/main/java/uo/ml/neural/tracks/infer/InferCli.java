package uo.ml.neural.tracks.infer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

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
import uo.ml.neural.tracks.core.preprocess.Padding;
import uo.ml.neural.tracks.core.preprocess.Resampler3D;
import uo.ml.neural.tracks.core.preprocess.ZScoreScaler;

/**
 * CLI tool for GPS track inference using trained neural network models.
 * Supports single model or ensemble predictions with uncertainty estimation.
 */
@Command(
    name = "tracks-infer",
    mixinStandardHelpOptions = true,
    version = "1.0.0-SNAPSHOT",
    description = "Performs inference on GPS tracks using trained neural network models"
)
public class InferCli implements Callable<Integer> {
    
    @Option(names = {"--model"}, required = true,
            description = "Path to trained model file (can be repeated for ensemble)")
    private List<Path> modelPaths = new ArrayList<>();
    
    @Option(names = {"--scaler"}, required = true,
            description = "Path to mu_sigma.json normalization parameters file")
    private Path scalerPath;
    
    @Option(names = {"--gpx"}, required = true,
            description = "Path to GPX file to process")
    private Path gpxPath;
    
    @Option(names = {"--step"}, defaultValue = "1.0",
            description = "Resampling step size in meters (default: ${DEFAULT-VALUE})")
    private double stepMeters;
    
    @Option(names = {"--filter"}, defaultValue = "none",
            description = "Altitude filter: median, sgolay, or none (default: ${DEFAULT-VALUE})")
    private FilterType filter;
    
    @Option(names = {"--maxlen"}, defaultValue = "5000",
            description = "Maximum sequence length for padding (default: ${DEFAULT-VALUE})")
    private int maxLength;
    
    public enum FilterType {
        median, sgolay, none
    }
    
    public static void main(String[] args) {
        int exitCode = new CommandLine(new InferCli()).execute(args);
        System.exit(exitCode);
    }
    
    @Override
    public Integer call() throws Exception {
        System.out.println("GPS Track Inference");
        System.out.println("==================");
        System.out.printf("GPX file: %s%n", gpxPath);
        System.out.printf("Models: %s%n", modelPaths);
        System.out.printf("Scaler: %s%n", scalerPath);
        System.out.printf("Step size: %.1f meters%n", stepMeters);
        System.out.printf("Filter: %s%n", filter);
        System.out.printf("Max length: %d%n", maxLength);
        System.out.println();
        
        // Validate inputs
        if (!Files.exists(gpxPath)) {
            System.err.println("Error: GPX file does not exist: " + gpxPath);
            return 1;
        }
        
        if (!Files.exists(scalerPath)) {
            System.err.println("Error: Scaler file does not exist: " + scalerPath);
            return 1;
        }
        
        for (Path modelPath : modelPaths) {
            if (!Files.exists(modelPath)) {
                System.err.println("Error: Model file does not exist: " + modelPath);
                return 1;
            }
        }
        
        try {
            // Process GPX track
            System.out.println("Processing GPX track...");
            TrackFeatures trackFeatures = processTrack();
            
            // Load models and make predictions
            System.out.printf("Loading %d model(s)...%n", modelPaths.size());
            List<double[]> predictions = new ArrayList<>();
            
            for (int i = 0; i < modelPaths.size(); i++) {
                Path modelPath = modelPaths.get(i);
                System.out.printf("Loading model %d/%d: %s%n", i + 1, modelPaths.size(), modelPath);
                
                MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath.toFile());
                double[] prediction = predict(model, trackFeatures);
                predictions.add(prediction);
                
                System.out.printf("Model %d prediction: [%.1f, %.1f, %.1f]%n", 
                    i + 1, prediction[0], prediction[1], prediction[2]);
            }
            
            // Compute final results
            InferenceResult result = computeResults(predictions);
            
            // Output results
            outputResults(result);
            
            System.out.println("Inference completed successfully!");
            return 0;
            
        } catch (Exception e) {
            System.err.println("Error during inference: " + e.getMessage());
            e.printStackTrace();
            return 1;
        }
    }
    
    private TrackFeatures processTrack() throws IOException {
        // Read GPX points
        List<GpxPoint> gpxPoints = GpxUtils.readGpx(gpxPath);
        if (gpxPoints.size() < 2) {
            throw new IOException("Track has fewer than 2 points");
        }
        
        System.out.printf("Loaded %d GPS points%n", gpxPoints.size());
        
        // Convert to UTM
        ProjectionUtils.UtmZone utmZone = ProjectionUtils.detectZone(gpxPoints);
        List<UtmPoint> utmPoints = ProjectionUtils.toUtm(gpxPoints, utmZone);
        
        System.out.printf("Converted to UTM zone: %s%n", utmZone);
        
        // Apply altitude filter if specified
        if (filter != FilterType.none) {
            utmPoints = applyAltitudeFilter(utmPoints);
            System.out.printf("Applied %s altitude filter%n", filter);
        }
        
        // Resample by 3D arc length
        List<UtmPoint> resampledPoints = Resampler3D.resampleByArcLength3D(utmPoints, stepMeters);
        System.out.printf("Resampled to %d points (step: %.1fm)%n", resampledPoints.size(), stepMeters);
        
        // Extract features
        List<SegmentFeature> features = FeatureExtractor.computeFeatures(resampledPoints);
        System.out.printf("Extracted %d segment features%n", features.size());
        
        return new TrackFeatures(features);
    }
    
    private List<UtmPoint> applyAltitudeFilter(List<UtmPoint> points) {
        // Extract altitude values
        List<Double> altitudes = points.stream()
            .map(UtmPoint::getZ)
            .toList();
        
        // Apply filter
        List<Double> filteredAltitudes = switch (filter) {
            case median -> Filters.medianFilter(altitudes, 5);
            case sgolay -> Filters.savitzkyGolay(altitudes, 5, 2);
            case none -> altitudes;
        };
        
        // Create new points with filtered altitudes
        List<UtmPoint> filtered = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            UtmPoint original = points.get(i);
            filtered.add(new UtmPoint(original.getE(), original.getN(), filteredAltitudes.get(i)));
        }
        
        return filtered;
    }
    
    private double[] predict(MultiLayerNetwork model, TrackFeatures trackFeatures) throws IOException {
        // Load Z-score scaler
        ZScoreScaler scaler = ZScoreScaler.load(scalerPath);
        
        // Apply normalization
        List<SegmentFeature> normalizedFeatures = trackFeatures.features.stream()
            .map(scaler::transform)
            .toList();
        
        // Pad features and create mask
        double[][] paddedFeatures = Padding.padFeatures(normalizedFeatures, maxLength);
        double[] mask = Padding.makeMask(normalizedFeatures.size(), maxLength);
        
        // Create INDArrays [1, nFeatures, time]
        INDArray features = Nd4j.zeros(1, 3, maxLength);
        INDArray featuresMask = Nd4j.zeros(1, maxLength);
        
        // Fill arrays
        for (int t = 0; t < maxLength; t++) {
            features.putScalar(new int[]{0, 0, t}, paddedFeatures[t][0]); // dh
            features.putScalar(new int[]{0, 1, t}, paddedFeatures[t][1]); // dz
            features.putScalar(new int[]{0, 2, t}, paddedFeatures[t][2]); // slope
            featuresMask.putScalar(new int[]{0, t}, mask[t]);
        }
        
        // Create dataset and predict
        DataSet dataSet = new DataSet(features, null, featuresMask, null);
        INDArray predictions = model.output(dataSet.getFeatures(), false);
        
        // Extract results
        return new double[]{
            predictions.getDouble(0, 0), // dist_total
            predictions.getDouble(0, 1), // desn_pos
            predictions.getDouble(0, 2)  // desn_neg
        };
    }
    
    private InferenceResult computeResults(List<double[]> predictions) {
        if (predictions.size() == 1) {
            // Single model - no uncertainty
            double[] pred = predictions.get(0);
            return new InferenceResult(pred[0], pred[1], pred[2], null);
        }
        
        // Ensemble - compute mean and standard deviation
        double[] means = new double[3];
        double[] stds = new double[3];
        
        // Compute means
        for (double[] pred : predictions) {
            for (int i = 0; i < 3; i++) {
                means[i] += pred[i];
            }
        }
        for (int i = 0; i < 3; i++) {
            means[i] /= predictions.size();
        }
        
        // Compute standard deviations
        for (double[] pred : predictions) {
            for (int i = 0; i < 3; i++) {
                stds[i] += Math.pow(pred[i] - means[i], 2);
            }
        }
        for (int i = 0; i < 3; i++) {
            stds[i] = Math.sqrt(stds[i] / predictions.size());
        }
        
        UncertaintyEstimate uncertainty = new UncertaintyEstimate(stds[0], stds[1], stds[2]);
        return new InferenceResult(means[0], means[1], means[2], uncertainty);
    }
    
    private void outputResults(InferenceResult result) throws IOException {
        // Text output
        System.out.println();
        System.out.println("=== INFERENCE RESULTS ===");
        System.out.printf("dist_total_m: %.1f%n", result.distTotal);
        System.out.printf("desnivel_pos_m: %.1f%n", result.desnivelPos);
        System.out.printf("desnivel_neg_m: %.1f%n", result.desnivelNeg);
        
        if (result.uncertainty != null) {
            System.out.println("uncertainty_sigma: {");
            System.out.printf("  dist: %.1f%n", result.uncertainty.dist);
            System.out.printf("  up: %.1f%n", result.uncertainty.up);
            System.out.printf("  down: %.1f%n", result.uncertainty.down);
            System.out.println("}");
        }
        
        // JSON output
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode jsonResult = mapper.createObjectNode();
        
        jsonResult.put("dist_total_m", Math.round(result.distTotal * 10) / 10.0);
        jsonResult.put("desnivel_pos_m", Math.round(result.desnivelPos * 10) / 10.0);
        jsonResult.put("desnivel_neg_m", Math.round(result.desnivelNeg * 10) / 10.0);
        
        if (result.uncertainty != null) {
            ObjectNode uncertaintyNode = mapper.createObjectNode();
            uncertaintyNode.put("dist", Math.round(result.uncertainty.dist * 10) / 10.0);
            uncertaintyNode.put("up", Math.round(result.uncertainty.up * 10) / 10.0);
            uncertaintyNode.put("down", Math.round(result.uncertainty.down * 10) / 10.0);
            jsonResult.set("uncertainty_sigma", uncertaintyNode);
        }
        
        // Save JSON to file
        String gpxBaseName = getBaseName(gpxPath);
        Path jsonOutputPath = Paths.get(gpxBaseName + "_inference.json");
        mapper.writerWithDefaultPrettyPrinter().writeValue(jsonOutputPath.toFile(), jsonResult);
        
        System.out.println();
        System.out.printf("Results saved to: %s%n", jsonOutputPath);
        
        // Also print JSON to stdout
        System.out.println();
        System.out.println("=== JSON OUTPUT ===");
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(jsonResult));
    }
    
    private String getBaseName(Path file) {
        String fileName = file.getFileName().toString();
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
    }
    
    private static class TrackFeatures {
        final List<SegmentFeature> features;
        
        TrackFeatures(List<SegmentFeature> features) {
            this.features = features;
        }
    }
    
    private static class InferenceResult {
        final double distTotal;
        final double desnivelPos;
        final double desnivelNeg;
        final UncertaintyEstimate uncertainty;
        
        InferenceResult(double distTotal, double desnivelPos, double desnivelNeg, 
                       UncertaintyEstimate uncertainty) {
            this.distTotal = distTotal;
            this.desnivelPos = desnivelPos;
            this.desnivelNeg = desnivelNeg;
            this.uncertainty = uncertainty;
        }
    }
    
    private static class UncertaintyEstimate {
        final double dist;
        final double up;
        final double down;
        
        UncertaintyEstimate(double dist, double up, double down) {
            this.dist = dist;
            this.up = up;
            this.down = down;
        }
    }
}