package uo.ml.neural.tracks.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.train.data.SequenceDataset;
import uo.ml.neural.tracks.train.eval.Baseline;
import uo.ml.neural.tracks.train.model.ModelFactory;

/**
 * Leave-One-Family-Out (LOFO) cross-validation trainer for GPS track analysis.
 * Trains models excluding one family at a time and evaluates on the excluded family.
 */
@Command(
    name = "lofo-trainer",
    mixinStandardHelpOptions = true,
    version = "1.0.0-SNAPSHOT",
    description = "Performs Leave-One-Family-Out cross-validation training"
)
public class LofoTrainerCli implements Callable<Integer> {
    
    @Option(names = {"--data"}, required = true,
            description = "Processed data directory (output of preprocessing)")
    private Path dataDir;
    
    @Option(names = {"--epochs"}, defaultValue = "100",
            description = "Maximum number of training epochs (default: ${DEFAULT-VALUE})")
    private int maxEpochs;
    
    @Option(names = {"--lr"}, defaultValue = "0.001", 
            description = "Learning rate (default: ${DEFAULT-VALUE})")
    private double learningRate;
    
    public static void main(String[] args) {
        int exitCode = new CommandLine(new LofoTrainerCli()).execute(args);
        System.exit(exitCode);
    }
    
    @Override
    public Integer call() throws Exception {
        System.out.println("Leave-One-Family-Out Cross-Validation");
        System.out.println("=====================================");
        System.out.printf("Data directory: %s%n", dataDir);
        System.out.printf("Max epochs: %d%n", maxEpochs);
        System.out.printf("Learning rate: %.4f%n", learningRate);
        System.out.println();
        
        // Validate data directory
        if (!Files.exists(dataDir) || !Files.isDirectory(dataDir)) {
            System.err.println("Error: Data directory does not exist: " + dataDir);
            return 1;
        }
        
        // Find all families
        List<String> allFamilies = findAllFamilies();
        if (allFamilies.isEmpty()) {
            System.err.println("Error: No families found in data directory");
            return 1;
        }
        
        System.out.printf("Found %d families for LOFO validation%n", allFamilies.size());
        System.out.println("Families: " + String.join(", ", allFamilies));
        System.out.println();
        
        // Perform LOFO validation
        List<FoldResult> results = new ArrayList<>();
        
        for (int fold = 0; fold < allFamilies.size(); fold++) {
            String testFamily = allFamilies.get(fold);
            List<String> trainFamilies = new ArrayList<>(allFamilies);
            trainFamilies.remove(testFamily);
            
            System.out.printf("=== FOLD %d/%d: Testing on %s ====%n", 
                fold + 1, allFamilies.size(), testFamily);
            System.out.printf("Training families: %s%n", String.join(", ", trainFamilies));
            
            try {
                FoldResult result = performFold(trainFamilies, Arrays.asList(testFamily));
                results.add(result);
                
                System.out.printf("Neural Network MAE: [%.3f, %.3f, %.3f] (overall: %.3f)%n",
                    result.nnMAE[0], result.nnMAE[1], result.nnMAE[2], result.nnOverallMAE);
                System.out.printf("Baseline MAE: [%.3f, %.3f, %.3f] (overall: %.3f)%n",
                    result.baselineMAE[0], result.baselineMAE[1], result.baselineMAE[2], result.baselineOverallMAE);
                System.out.println();
                
            } catch (Exception e) {
                System.err.printf("Error in fold %d (family %s): %s%n", fold + 1, testFamily, e.getMessage());
                e.printStackTrace();
                return 1;
            }
        }
        
        // Compute and report overall statistics
        reportOverallResults(results);
        
        System.out.println("LOFO validation completed successfully!");
        return 0;
    }
    
    private List<String> findAllFamilies() throws IOException {
        Path featuresDir = dataDir.resolve("features");
        if (!Files.exists(featuresDir)) {
            return new ArrayList<>();
        }
        
        try (var stream = Files.list(featuresDir)) {
            return stream
                .filter(Files::isDirectory)
                .map(p -> p.getFileName().toString())
                .sorted()
                .collect(Collectors.toList());
        }
    }
    
    private FoldResult performFold(List<String> trainFamilies, List<String> testFamilies) throws IOException {
        // Load training data (excluding test families)
        SequenceDataset trainData = SequenceDataset.load(dataDir, testFamilies);
        
        // Load test data (only test families)
        SequenceDataset testData = SequenceDataset.load(dataDir, trainFamilies);
        
        System.out.printf("Training samples: %d, Test samples: %d%n", 
            trainData.getBatchSize(), testData.getBatchSize());
        
        // Create and train neural network model
        MultiLayerNetwork model = ModelFactory.createLSTMModel(trainData.getNumFeatures(), learningRate);
        
        // Create training dataset
        DataSet trainingSet = new DataSet(trainData.getFeatures(), trainData.getLabels(), 
                                         trainData.getFeaturesMask(), null);
        
        // Simple training loop (could add early stopping here)
        System.out.print("Training progress: ");
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            model.fit(trainingSet);
            
            if (epoch % (maxEpochs / 10) == 0) {
                System.out.print(".");
            }
        }
        System.out.println(" Done");
        
        // Evaluate neural network on test set
        DataSet testSet = new DataSet(testData.getFeatures(), testData.getLabels(), 
                                     testData.getFeaturesMask(), null);
        INDArray predictions = model.output(testSet.getFeatures(), false);
        double[] nnMAE = computeMAE(predictions, testData.getLabels());
        double nnOverallMAE = (nnMAE[0] + nnMAE[1] + nnMAE[2]) / 3.0;
        
        // Evaluate baseline on test set  
        double[] baselineMAE = evaluateBaseline(testData);
        double baselineOverallMAE = (baselineMAE[0] + baselineMAE[1] + baselineMAE[2]) / 3.0;
        
        return new FoldResult(testFamilies.get(0), nnMAE, nnOverallMAE, baselineMAE, baselineOverallMAE);
    }
    
    private double[] computeMAE(INDArray predictions, INDArray labels) {
        int batchSize = (int) predictions.size(0);
        double[] totalMAE = new double[3];
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < 3; j++) {
                double pred = predictions.getDouble(i, j);
                double label = labels.getDouble(i, j);
                totalMAE[j] += Math.abs(pred - label);
            }
        }
        
        // Average over batch
        for (int j = 0; j < 3; j++) {
            totalMAE[j] /= batchSize;
        }
        
        return totalMAE;
    }
    
    private double[] evaluateBaseline(SequenceDataset testData) {
        // For baseline evaluation, we need to reconstruct the original features
        // Since we don't have direct access to unnormalized features, we'll approximate
        // by using the normalized features for relative calculations
        
        int batchSize = testData.getBatchSize();
        double[] totalMAE = new double[3];
        
        // This is a simplified baseline evaluation using normalized features
        // In a real implementation, you might want to store unnormalized features
        // or reverse the normalization
        
        for (int i = 0; i < batchSize; i++) {
            // Extract features for this sample
            List<SegmentFeature> features = extractFeaturesFromArray(testData.getFeatures(), i);
            
            // Compute baseline prediction
            double[] baseline = Baseline.computeBaseline(features);
            
            // Get true labels
            double[] trueLabels = new double[]{
                testData.getLabels().getDouble(i, 0),
                testData.getLabels().getDouble(i, 1), 
                testData.getLabels().getDouble(i, 2)
            };
            
            // Compute MAE for this sample
            double[] sampleMAE = Baseline.computeMAE(baseline, trueLabels);
            for (int j = 0; j < 3; j++) {
                totalMAE[j] += sampleMAE[j];
            }
        }
        
        // Average over batch
        for (int j = 0; j < 3; j++) {
            totalMAE[j] /= batchSize;
        }
        
        return totalMAE;
    }
    
    private List<SegmentFeature> extractFeaturesFromArray(INDArray features, int sampleIndex) {
        List<SegmentFeature> result = new ArrayList<>();
        int maxLength = (int) features.size(2);
        
        for (int t = 0; t < maxLength; t++) {
            double dh = features.getDouble(sampleIndex, 0, t);
            double dz = features.getDouble(sampleIndex, 1, t);
            double slope = features.getDouble(sampleIndex, 2, t);
            
            // Stop at padding (assuming zero-padding)
            if (dh == 0.0 && dz == 0.0 && slope == 0.0) {
                break;
            }
            
            result.add(new SegmentFeature(dh, dz, slope));
        }
        
        return result;
    }
    
    private void reportOverallResults(List<FoldResult> results) {
        System.out.println("=== OVERALL RESULTS ===");
        
        // Compute means and standard deviations
        double[] nnMeans = new double[4]; // 3 components + overall
        double[] nnStds = new double[4];
        double[] baselineMeans = new double[4];
        double[] baselineStds = new double[4];
        
        // Compute means
        for (FoldResult result : results) {
            for (int i = 0; i < 3; i++) {
                nnMeans[i] += result.nnMAE[i];
                baselineMeans[i] += result.baselineMAE[i];
            }
            nnMeans[3] += result.nnOverallMAE;
            baselineMeans[3] += result.baselineOverallMAE;
        }
        
        for (int i = 0; i < 4; i++) {
            nnMeans[i] /= results.size();
            baselineMeans[i] /= results.size();
        }
        
        // Compute standard deviations
        for (FoldResult result : results) {
            for (int i = 0; i < 3; i++) {
                nnStds[i] += Math.pow(result.nnMAE[i] - nnMeans[i], 2);
                baselineStds[i] += Math.pow(result.baselineMAE[i] - baselineMeans[i], 2);
            }
            nnStds[3] += Math.pow(result.nnOverallMAE - nnMeans[3], 2);
            baselineStds[3] += Math.pow(result.baselineOverallMAE - baselineMeans[3], 2);
        }
        
        for (int i = 0; i < 4; i++) {
            nnStds[i] = Math.sqrt(nnStds[i] / results.size());
            baselineStds[i] = Math.sqrt(baselineStds[i] / results.size());
        }
        
        // Report results
        System.out.printf("Neural Network MAE (mean ± std):%n");
        System.out.printf("  Distance:  %.3f ± %.3f%n", nnMeans[0], nnStds[0]);
        System.out.printf("  Elev (+):  %.3f ± %.3f%n", nnMeans[1], nnStds[1]);
        System.out.printf("  Elev (-):  %.3f ± %.3f%n", nnMeans[2], nnStds[2]);
        System.out.printf("  Overall:   %.3f ± %.3f%n", nnMeans[3], nnStds[3]);
        System.out.println();
        
        System.out.printf("Baseline MAE (mean ± std):%n");
        System.out.printf("  Distance:  %.3f ± %.3f%n", baselineMeans[0], baselineStds[0]);
        System.out.printf("  Elev (+):  %.3f ± %.3f%n", baselineMeans[1], baselineStds[1]);
        System.out.printf("  Elev (-):  %.3f ± %.3f%n", baselineMeans[2], baselineStds[2]);
        System.out.printf("  Overall:   %.3f ± %.3f%n", baselineMeans[3], baselineStds[3]);
        System.out.println();
        
        // Individual fold results
        System.out.println("Results by family:");
        System.out.println("Family\t\tNeural Network\t\tBaseline");
        System.out.println("------\t\t--------------\t\t--------");
        for (FoldResult result : results) {
            System.out.printf("%-12s\t%.3f\t\t\t%.3f%n", 
                result.testFamily, result.nnOverallMAE, result.baselineOverallMAE);
        }
    }
    
    private static class FoldResult {
        final String testFamily;
        final double[] nnMAE;
        final double nnOverallMAE;
        final double[] baselineMAE;
        final double baselineOverallMAE;
        
        FoldResult(String testFamily, double[] nnMAE, double nnOverallMAE, 
                  double[] baselineMAE, double baselineOverallMAE) {
            this.testFamily = testFamily;
            this.nnMAE = nnMAE.clone();
            this.nnOverallMAE = nnOverallMAE;
            this.baselineMAE = baselineMAE.clone();
            this.baselineOverallMAE = baselineOverallMAE;
        }
    }
}