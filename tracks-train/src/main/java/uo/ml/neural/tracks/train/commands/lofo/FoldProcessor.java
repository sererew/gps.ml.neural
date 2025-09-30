package uo.ml.neural.tracks.train.commands.lofo;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
			Path dataDir) {

		// Load datasets
		SequenceDataset trainData = SequenceDataset.load(dataDir, trainFamilies);
		SequenceDataset testData = SequenceDataset.load(dataDir, testFamilies);

		System.out.printf("Training samples: %d, Test samples: %d%n",
				trainData.getBatchSize(), testData.getBatchSize());

		// Train model
		MultiLayerNetwork model = trainModel(trainData);

		// Evaluate neural network and capture predictions
		EvaluationResult nnResult = evaluateNeuralNetworkWithPredictions(model, testData);
		double[] nnMAE = nnResult.mae;
		double nnOverallMAE = computeOverallMAE(nnMAE);

		// Evaluate baseline
		double[] baselineMAE = evaluator.evaluateBaseline(testData);
		double baselineOverallMAE = computeOverallMAE(baselineMAE);

		// Extract predictions and actual labels for saving
		Map<String, double[]> predictions = extractPredictionsMap(nnResult.predictions, testData);
		Map<String, double[]> actualLabels = extractActualLabelsMap(testData);

		return new FoldResult(
				testFamilies.get(0), 
				trainFamilies,
				nnMAE, nnOverallMAE,
				baselineMAE, baselineOverallMAE,
				model,  // Include trained model
				predictions,  // Include predictions
				actualLabels  // Include actual labels
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
			System.out.print(".");	// Progress indicator
		}
		System.out.println(" Done");

		return model;
	}

	private EvaluationResult evaluateNeuralNetworkWithPredictions(
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
		double[] mae = evaluator.computeMAE(predictions, testData.getLabels());
		
		return new EvaluationResult(mae, predictions);
	}

	private double computeOverallMAE(double[] mae) {
		return (mae[0] + mae[1] + mae[2]) / 3.0;
	}

	private Map<String, double[]> extractPredictionsMap(INDArray predictions, SequenceDataset testData) {
		Map<String, double[]> predictionsMap = new HashMap<>();
		
		// Get track names from test data
		List<String> trackNames = testData.getTrackNames();
		
		for (int i = 0; i < predictions.size(0); i++) {
			String trackName = trackNames.get(i);
			double[] predictionArray = new double[3];
			for (int j = 0; j < 3; j++) {
				predictionArray[j] = predictions.getDouble(i, j);
			}
			predictionsMap.put(trackName, predictionArray);
		}
		
		return predictionsMap;
	}

	private Map<String, double[]> extractActualLabelsMap(SequenceDataset testData) {
		Map<String, double[]> labelsMap = new HashMap<>();
		INDArray labels = testData.getLabels();
		List<String> trackNames = testData.getTrackNames();
		
		for (int i = 0; i < labels.size(0); i++) {
			String trackName = trackNames.get(i);
			double[] labelArray = new double[3];
			for (int j = 0; j < 3; j++) {
				labelArray[j] = labels.getDouble(i, j);
			}
			labelsMap.put(trackName, labelArray);
		}
		
		return labelsMap;
	}

	private static class EvaluationResult {
		final double[] mae;
		final INDArray predictions;

		EvaluationResult(double[] mae, INDArray predictions) {
			this.mae = mae;
			this.predictions = predictions;
		}
	}
}