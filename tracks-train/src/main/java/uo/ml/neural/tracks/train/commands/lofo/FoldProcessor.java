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
		EvaluationResult nnResult = evaluateModel(model, testData);
		float[] nnMAE = nnResult.mae;
		float nnOverallMAE = computeOverallMAE(nnMAE);

		// Evaluate baseline
		float[] baselineMAE = evaluator.evaluateBaseline(testData);
		float baselineOverallMAE = computeOverallMAE(baselineMAE);

		// Extract predictions and actual labels for saving
		Map<String, float[]> predictions = extractPredictionsMap(nnResult.predictions, testData);
		Map<String, float[]> actualLabels = extractActualLabelsMap(testData);

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
		long startTime = System.currentTimeMillis();
		for (int epoch = 0; epoch < 1/*maxEpochs*/; epoch++) {
			model.fit(trainingSet);
			System.out.printf("%.1f min%n",
					(System.currentTimeMillis() - startTime) / 1000 / 60.0
				);	// Progress indicator
			startTime = System.currentTimeMillis();
//			System.out.print(".");	// Progress indicator
		}
		System.out.println("Training done");

		return model;
	}

	private EvaluationResult evaluateModel(
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
		float[] mae = evaluator.computeMAE(predictions, testData.getLabels());
		
		return new EvaluationResult(mae, predictions);
	}

	private float computeOverallMAE(float[] mae) {
		return (mae[0] + mae[1] + mae[2]) / 3.0f;
	}

	private Map<String, float[]> extractPredictionsMap(INDArray predictions, SequenceDataset testData) {
		Map<String, float[]> predictionsMap = new HashMap<>();
		
		// Get track names from test data
		List<String> trackNames = testData.getTrackNames();
		
		for (int i = 0; i < predictions.size(0); i++) {
			String trackName = trackNames.get(i);
			float[] predictionArray = new float[3];
			for (int j = 0; j < 3; j++) {
				predictionArray[j] = predictions.getFloat(i, j);
			}
			predictionsMap.put(trackName, predictionArray);
		}
		
		return predictionsMap;
	}

	private Map<String, float[]> extractActualLabelsMap(SequenceDataset testData) {
		Map<String, float[]> labelsMap = new HashMap<>();
		INDArray labels = testData.getLabels();
		List<String> trackNames = testData.getTrackNames();
		
		for (int i = 0; i < labels.size(0); i++) {
			String trackName = trackNames.get(i);
			float[] labelArray = new float[3];
			for (int j = 0; j < 3; j++) {
				labelArray[j] = labels.getFloat(i, j);
			}
			labelsMap.put(trackName, labelArray);
		}
		
		return labelsMap;
	}

	private static record EvaluationResult (
				float[] mae,
				INDArray predictions
			){}
}