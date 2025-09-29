package uo.ml.neural.tracks.train.commands.lofo;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import uo.ml.neural.tracks.train.data.SequenceDataset;
import uo.ml.neural.tracks.train.model.FoldResult;
import uo.ml.neural.tracks.train.model.ModelFactory;

/**
 * Handles the training and evaluation of a single fold in LOFO
 * cross-validation.
 */
public class FoldProcessor {

	private final ModelEvaluator evaluator = new ModelEvaluator();
	private final int maxEpochs;
	private final double learningRate;

	public FoldProcessor(int maxEpochs, double learningRate) {
		this.maxEpochs = maxEpochs;
		this.learningRate = learningRate;
	}

	/**
	 * Processes a single fold: trains model on training families and evaluates
	 * on test family.
	 */
	public FoldResult process(
			List<String> trainFamilies,
			List<String> testFamilies, 
			Path dataDir) throws IOException {

		// Load datasets
		SequenceDataset trainData = SequenceDataset.load(dataDir, testFamilies);
		SequenceDataset testData = SequenceDataset.load(dataDir, trainFamilies);

		System.out.printf("Training samples: %d, Test samples: %d%n",
				trainData.getBatchSize(), testData.getBatchSize());

		// Train model
		MultiLayerNetwork model = trainModel(trainData);

		// Evaluate neural network
		double[] nnMAE = evaluateNeuralNetwork(model, testData);
		double nnOverallMAE = computeOverallMAE(nnMAE);

		// Evaluate baseline
		double[] baselineMAE = evaluator.evaluateBaseline(testData);
		double baselineOverallMAE = computeOverallMAE(baselineMAE);

		return new FoldResult(
				testFamilies.get(0), 
				nnMAE, nnOverallMAE,
				baselineMAE, baselineOverallMAE
			);
	}

	private MultiLayerNetwork trainModel(SequenceDataset trainData) {
		// Create model
		MultiLayerNetwork model = ModelFactory
				.createLSTMModel(trainData.getNumFeatures(), learningRate);

		// Create training dataset
		DataSet trainingSet = new DataSet(
				trainData.getFeatures(),
				trainData.getLabels(), 
				trainData.getFeaturesMask(), 
				null	// No label mask
			);

		// Training loop
		System.out.print("Training progress: ");
		for (int epoch = 0; epoch < maxEpochs; epoch++) {
			model.fit(trainingSet);

			if (epoch % (maxEpochs / 10) == 0) {
				System.out.print(".");	// Progress indicator one . per 10%
			}
		}
		System.out.println(" Done");

		return model;
	}

	private double[] evaluateNeuralNetwork(
			MultiLayerNetwork model,
			SequenceDataset testData) {
		
		DataSet testSet = new DataSet(
					testData.getFeatures(),
					testData.getLabels(), 
					testData.getFeaturesMask(), 
					null	// No label mask
				);
		
		// Get predictions
		INDArray predictions = model.output(testSet.getFeatures(), false);
		return evaluator.computeMAE(predictions, testData.getLabels());
	}

	private double computeOverallMAE(double[] mae) {
		return (mae[0] + mae[1] + mae[2]) / 3.0;
	}
}