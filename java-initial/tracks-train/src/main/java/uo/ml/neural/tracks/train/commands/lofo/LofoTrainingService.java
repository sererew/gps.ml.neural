package uo.ml.neural.tracks.train.commands.lofo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.core.exception.IO;
import uo.ml.neural.tracks.train.model.FoldResult;

/**
 * Service for performing Leave-One-Family-Out (LOFO) cross-validation for GPS
 * track analysis models.
 */
public class LofoTrainingService {

	private final Path dataDir;
	private final Path outputDir;
	private final int maxEpochs;
	private final double learningRate;

	public LofoTrainingService(
			Path dataDir, Path outputDir, 
			int maxEpochs, double learningRate) {
		
		this.dataDir = dataDir;
		this.outputDir = outputDir;
		this.maxEpochs = maxEpochs;
		this.learningRate = learningRate;
	}

	public void performLofoTraining() {
		printHeader();

		validateDataDirectory();

		List<String> allFamilies = findAllFamilies();
		if (allFamilies.isEmpty()) {
			throw new CommandException("No families found in data directory");
		}

		printFamiliesInfo(allFamilies);

		List<FoldResult> results = executeLofoFolds(allFamilies);
		new ResultsReporter().reportOverallResults(results);

		// Save results to output directory
		new LofoResultsSaver(outputDir).saveResults(results);

		System.out.println("LOFO validation completed successfully!");
	}

	private void printHeader() {
		System.out.println("Leave-One-Family-Out Cross-Validation");
		System.out.println("=====================================");
		System.out.printf("Data directory: %s%n", dataDir);
		System.out.printf("Output directory: %s%n", outputDir);
		System.out.printf("Max epochs: %d%n", maxEpochs);
		System.out.printf("Learning rate: %.4f%n", learningRate);
		System.out.println();
	}

	private void validateDataDirectory() {
		if (!Files.exists(dataDir) || !Files.isDirectory(dataDir)) {
			throw new CommandException(
				"Data directory does not exist: " + dataDir);
		}
	}

	private void printFamiliesInfo(List<String> allFamilies) {
		System.out.printf(
				"Found %d families for LOFO validation%n",
				allFamilies.size()
			);
		System.out.println("Families: " + String.join(", ", allFamilies));
		System.out.println();
	}

	private List<FoldResult> executeLofoFolds(List<String> allFamilies) {
		List<FoldResult> results = new ArrayList<>();
		FoldProcessor foldProcessor = new FoldProcessor(maxEpochs, learningRate);

		for (int fold = 0; fold < allFamilies.size(); fold++) {
			String testFamily = allFamilies.get(fold);
			List<String> trainFamilies = createTrainFamilies(
					allFamilies,
					testFamily
				);

			printFoldHeader(fold, allFamilies.size(), testFamily, trainFamilies);

			FoldResult result = foldProcessor.process(
					trainFamilies,
					testFamily, 
					dataDir
				);

			results.add(result);
			printFoldResults(result);
		}

		return results;
	}

	private List<String> createTrainFamilies(List<String> allFamilies,
			String testFamily) {
		List<String> trainFamilies = new ArrayList<>(allFamilies);
		trainFamilies.remove(testFamily);
		return trainFamilies;
	}

	private void printFoldHeader(int fold, int totalFolds, String testFamily,
			List<String> trainFamilies) {
		System.out.printf("=== FOLD %d/%d: Testing on %s ====%n", 
				fold + 1,
				totalFolds, 
				testFamily
			);
		System.out.printf("Training families: %s%n",
				String.join(", ", trainFamilies)
			);
	}

	private void printFoldResults(FoldResult result) {
		float[] nnMAE = result.getNnMAE();

		System.out.printf(Locale.US,
				"Neural Network MAE: [%.3f, %.3f, %.3f] (overall: %.3f)%n",
				nnMAE[0], nnMAE[1], nnMAE[2], result.getNnOverallMAE());
		System.out.println();
	}

	private List<String> findAllFamilies() {
		Path featuresDir = dataDir.resolve("features");
		if (!Files.exists(featuresDir)) {
			return new ArrayList<>();
		}

		return IO.get(() -> Files.list(featuresDir)
					.filter(Files::isDirectory)
					.map(p -> p.getFileName().toString())
					.sorted()
					.toList()
				);
	}
}