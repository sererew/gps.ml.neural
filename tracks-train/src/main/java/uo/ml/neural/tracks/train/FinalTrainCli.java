package uo.ml.neural.tracks.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.Callable;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import uo.ml.neural.tracks.train.data.SequenceDataset;
import uo.ml.neural.tracks.train.model.ModelFactory;

/**
 * Final training CLI that trains a model on all available families
 * and saves the trained model and normalization parameters.
 */
@Command(
    name = "final-trainer",
    mixinStandardHelpOptions = true,
    version = "1.0.0-SNAPSHOT",
    description = "Trains final model on all families and saves for inference"
)
public class FinalTrainCli implements Callable<Integer> {
    
    @Option(names = {"--data"}, 
    		required = true,
            description = "Processed data directory (output of preprocessing)"
    )
    private Path dataDir;
    
    @Option(names = {"--out"}, 
    		required = true,
            description = "Output directory for trained model"
    )
    private Path outputDir;
    
    @Option(names = {"--epochs"}, 
    		defaultValue = "150",
            description = "Number of training epochs (default: ${DEFAULT-VALUE})"
    )
    private int epochs;
    
    @Option(names = {"--lr"}, 
    		defaultValue = "0.001",
            description = "Learning rate (default: ${DEFAULT-VALUE})"
    )
    private double learningRate;
    
    public static void main(String[] args) {
        int exitCode = new CommandLine(new FinalTrainCli()).execute(args);
        System.exit(exitCode);
    }
    
    @Override
    public Integer call() throws Exception {
        System.out.println("Final Model Training");
        System.out.println("===================");
        System.out.printf("Data directory: %s%n", dataDir);
        System.out.printf("Output directory: %s%n", outputDir);
        System.out.printf("Epochs: %d%n", epochs);
        System.out.printf("Learning rate: %.4f%n", learningRate);
        System.out.println();
        
        // Validate data directory
        if (!Files.exists(dataDir) || !Files.isDirectory(dataDir)) {
            System.err.println("Error: Data directory does not exist: " + dataDir);
            return 1;
        }
        
        // Create output directory
        Files.createDirectories(outputDir);
        
        try {
            // Load all training data
            System.out.println("Loading training data...");
            SequenceDataset trainData = SequenceDataset.load(dataDir);
            
            System.out.printf("Loaded %d training samples%n", trainData.getBatchSize());
            System.out.printf("Features: %d, Max sequence length: %d%n", 
                trainData.getNumFeatures(), trainData.getMaxSequenceLength());
            System.out.println();
            
            // Create model
            System.out.println("Creating neural network model...");
            MultiLayerNetwork model = ModelFactory.createLSTMModel(trainData.getNumFeatures(), learningRate);
            
            System.out.println("Model architecture:");
            System.out.println(model.summary());
            System.out.println();
            
            // Create training dataset
            DataSet trainingSet = new DataSet(trainData.getFeatures(), trainData.getLabels(),
                                             trainData.getFeaturesMask(), null);
            
            // Training loop with progress reporting
            System.out.println("Training model...");
            System.out.print("Progress: ");
            
            for (int epoch = 0; epoch < epochs; epoch++) {
                model.fit(trainingSet);
                
                // Print progress dots
                if (epoch % (epochs / 20) == 0) {
                    System.out.print(".");
                }
                
                // Print detailed progress every 10%
                if (epoch % (epochs / 10) == 0) {
                    double score = model.score();
                    System.out.printf(" [Epoch %d/%d, Loss: %.6f] ", epoch, epochs, score);
                }
            }
            
            System.out.println(" Training completed!");
            System.out.println();
            
            // Save model
            System.out.println("Saving model and configuration...");
            Path modelPath = outputDir.resolve("model.zip");
            ModelSerializer.writeModel(model, modelPath.toFile(), true);
            System.out.printf("Model saved to: %s%n", modelPath);
            
            // Copy normalization parameters
            Path sourceMuSigma = dataDir.resolve("mu_sigma.json");
            Path destMuSigma = outputDir.resolve("mu_sigma.json");
            
            if (Files.exists(sourceMuSigma)) {
                Files.copy(sourceMuSigma, destMuSigma, StandardCopyOption.REPLACE_EXISTING);
                System.out.printf("Normalization parameters copied to: %s%n", destMuSigma);
            } else {
                System.err.println("Warning: mu_sigma.json not found in data directory");
            }
            
            // Create README with usage instructions
            createReadme();
            System.out.printf("Usage instructions saved to: %s%n", outputDir.resolve("README.md"));
            
            System.out.println();
            System.out.println("Final training completed successfully!");
            System.out.println("Model is ready for inference.");
            
            return 0;
            
        } catch (Exception e) {
            System.err.println("Error during training: " + e.getMessage());
            e.printStackTrace();
            return 1;
        }
    }
    
    private void createReadme() throws IOException {
        String readmeContent = """
            # Trained GPS Tracks Model
            
            This directory contains a trained neural network model for GPS track analysis.
            
            ## Files
            
            - `model.zip`: Trained DL4J model (MultiLayerNetwork)
            - `mu_sigma.json`: Z-score normalization parameters
            - `README.md`: This file
            
            ## Model Architecture
            
            - LSTM layer (128 units, tanh activation)
            - Global pooling (LAST - uses mask for variable-length sequences)
            - Dense layer (64 units, ReLU activation)
            - Output layer (3 units, linear activation)
            
            ## Input/Output
            
            **Input**: Sequences of GPS track features
            - Feature 0: `dh` - 2D horizontal distance (normalized)
            - Feature 1: `dz` - Vertical elevation change (normalized)  
            - Feature 2: `slope` - Slope = dz/(dh+1e-6) (normalized)
            
            **Output**: 3 values per track
            - Output 0: Total horizontal distance traveled
            - Output 1: Total positive elevation gain
            - Output 2: Total negative elevation loss (absolute)
            
            ## Usage Commands
            
            ### LOFO Cross-Validation
            ```bash
            java -cp tracks-train-jar-with-dependencies.jar uo.ml.neural.tracks.train.LofoTrainerCli --data ./data/processed
            ```
            
            ### Final Model Training
            ```bash
            java -cp tracks-train-jar-with-dependencies.jar uo.ml.neural.tracks.train.FinalTrainCli --data ./data/processed --out ./model
            ```
            
            ### Model Inference (using tracks-infer module)
            ```bash
            java -jar tracks-infer-jar-with-dependencies.jar --model ./model --input track.gpx
            ```
            
            ## Training Parameters
            
            - Optimizer: Adam
            - Learning rate: """ + learningRate + """
            - Loss function: Mean Absolute Error
            - Training epochs: """ + epochs + """
            - Batch processing: All samples in single batch
            
            ## Performance
            
            The model was trained on all available family data. For performance metrics,
            run LOFO cross-validation using the LofoTrainerCli command above.
            """;
        
        Files.writeString(outputDir.resolve("README.md"), readmeContent);
    }
}