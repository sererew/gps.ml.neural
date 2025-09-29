package uo.ml.neural.tracks.train.commands.finaltrain;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;

import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.core.exception.IO;
import uo.ml.neural.tracks.train.data.SequenceDataset;
import uo.ml.neural.tracks.train.model.ModelFactory;

/**
 * Service for final training on all available families and saving the model.
 */
public class FinalTrainingService {
    private final Path dataDir;
    private final Path outputDir;
    private final int epochs;
    private final double learningRate;

    public FinalTrainingService(
            Path dataDir,
            Path outputDir,
            int epochs,
            double learningRate) {
    	
        this.dataDir = dataDir;
        this.outputDir = outputDir;
        this.epochs = epochs;
        this.learningRate = learningRate;
    }

    public void train() {
        printHeader();
        validateDirectories();
        
        // Load all training data
        System.out.println("Loading training data...");
        SequenceDataset trainData = SequenceDataset.load(dataDir);
        printDatasetInfo(trainData);
        
        // Create model
        System.out.println("Creating neural network model...");
        MultiLayerNetwork model = ModelFactory.createLSTMModel(
        		trainData.getNumFeatures(), 
        		learningRate
       		);
        printModelArchitecture(model);
        
        // Create training dataset
        DataSet trainingSet = new DataSet(
        		trainData.getFeatures(), 
        		trainData.getLabels(),
                trainData.getFeaturesMask(), 
                null	// No label mask needed
        	);
        
        // Training loop with progress reporting
        System.out.println("Training model...");
        System.out.print("Progress: ");

        trainingLoop(model, trainingSet);
        
        System.out.println(" Training completed!");
        
        // Save model
        System.out.println("Saving model and configuration...");
        saveModel(model);
        saveNormalizationParameters();
        
        System.out.println("%nFinal training completed successfully!");
        System.out.println("Model is ready for inference.");
   }

	private void trainingLoop(MultiLayerNetwork model, DataSet trainingSet) {
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
	}

	private void saveNormalizationParameters() {
		Path sourceMuSigma = dataDir.resolve("mu_sigma.json");
        Path destMuSigma = outputDir.resolve("mu_sigma.json");
        
        IO.shallow(() -> Files.copy(
        		sourceMuSigma, 
        		destMuSigma, 
        		StandardCopyOption.REPLACE_EXISTING
        ));
        System.out.printf("Normalization parameters copied to: %s%n", destMuSigma);
	}

	private void saveModel(MultiLayerNetwork model) {
		Path modelPath = outputDir.resolve("model.zip");
        IO.shallow(() -> ModelSerializer.writeModel(model, modelPath.toFile(), true));
        System.out.printf("Model saved to: %s%n", modelPath);
	}

	private void printModelArchitecture(MultiLayerNetwork model) {
		System.out.println("Model architecture:");
        System.out.println(model.summary());
        System.out.println();
	}

	private void printDatasetInfo(SequenceDataset trainData) {
		System.out.printf("Loaded %d training samples%n", trainData.getBatchSize());
        System.out.printf("Features: %d, Max sequence length: %d%n%n", 
        		trainData.getNumFeatures(), 
        		trainData.getMaxSequenceLength()
        	);
	}

    private void printHeader() {
        System.out.println("Final Model Training");
        System.out.println("====================");
        System.out.printf("Data directory: %s%n", dataDir);
        System.out.printf("Output directory: %s%n", outputDir);
        System.out.printf("Epochs: %d%n", epochs);
        System.out.printf("Learning rate: %.4f%n", learningRate);
        System.out.println();
    }

    private void validateDirectories() {
        if (!Files.exists(dataDir) || !Files.isDirectory(dataDir)) {
            throw new CommandException("Data directory does not exist or is not a directory: " + dataDir);
        }
        if (!Files.exists(outputDir)) {
            IO.shallow(() -> Files.createDirectories(outputDir));
        }
    }

}