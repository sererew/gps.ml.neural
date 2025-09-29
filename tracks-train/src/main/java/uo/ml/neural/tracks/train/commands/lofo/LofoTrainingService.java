package uo.ml.neural.tracks.train.commands.lofo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import uo.ml.neural.tracks.train.model.FoldResult;

/**
 * Service for performing Leave-One-Family-Out (LOFO) cross-validation for GPS
 * track analysis models.
 */
public class LofoTrainingService {

	private final Path dataDir;
	private final int maxEpochs;
	private final double learningRate;

	public LofoTrainingService(Path dataDir, int maxEpochs,
			double learningRate) {
		this.dataDir = dataDir;
		this.maxEpochs = maxEpochs;
		this.learningRate = learningRate;
	}

	public Integer performLofoValidation() throws Exception {
		printHeader();

		if (!validateDataDirectory()) {
			return 1;
		}

		List<String> allFamilies = findAllFamilies();
		if (allFamilies.isEmpty()) {
			System.err.println("Error: No families found in data directory");
			return 1;
		}

		printFamiliesInfo(allFamilies);

		List<FoldResult> results = executeLofoFolds(allFamilies);
		if (results == null) {
			return 1;
		}

		new ResultsReporter().reportOverallResults(results);

		System.out.println("LOFO validation completed successfully!");
		return 0;
	}

	private void printHeader() {
		System.out.println("Leave-One-Family-Out Cross-Validation");
		System.out.println("=====================================");
		System.out.printf("Data directory: %s%n", dataDir);
		System.out.printf("Max epochs: %d%n", maxEpochs);
		System.out.printf("Learning rate: %.4f%n", learningRate);
		System.out.println();
	}

	private boolean validateDataDirectory() {
		if (!Files.exists(dataDir) || !Files.isDirectory(dataDir)) {
			System.err.println(
					"Error: Data directory does not exist: " + dataDir);
			return false;
		}
		return true;
	}

	private void printFamiliesInfo(List<String> allFamilies) {
		System.out.printf("Found %d families for LOFO validation%n",
				allFamilies.size());
		System.out.println("Families: " + String.join(", ", allFamilies));
		System.out.println();
	}

	private List<FoldResult> executeLofoFolds(List<String> allFamilies) {
		List<FoldResult> results = new ArrayList<>();
		FoldProcessor foldProcessor = new FoldProcessor(maxEpochs,
				learningRate);

		for (int fold = 0; fold < allFamilies.size(); fold++) {
			String testFamily = allFamilies.get(fold);
			List<String> trainFamilies = createTrainFamilies(allFamilies,
					testFamily);

			printFoldHeader(fold, allFamilies.size(), testFamily,
					trainFamilies);

			try {
				FoldResult result = foldProcessor.process(trainFamilies,
						List.of(testFamily), dataDir);

				results.add(result);
				printFoldResults(result);

			} catch (Exception e) {
				System.err.printf("Error in fold %d (family %s): %s%n",
						fold + 1, testFamily, e.getMessage());
				e.printStackTrace();
				return null;
			}
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
		double[] nnMAE = result.getNnMAE();
		double[] baselineMAE = result.getBaselineMAE();

		System.out.printf(
				"Neural Network MAE: [%.3f, %.3f, %.3f] (overall: %.3f)%n",
				nnMAE[0], nnMAE[1], nnMAE[2], result.getNnOverallMAE());
		System.out.printf("Baseline MAE: [%.3f, %.3f, %.3f] (overall: %.3f)%n",
				baselineMAE[0], baselineMAE[1], baselineMAE[2],
				result.getBaselineOverallMAE());
		System.out.println();
	}

	private List<String> findAllFamilies() throws IOException {
		Path featuresDir = dataDir.resolve("features");
		if (!Files.exists(featuresDir)) {
			return new ArrayList<>();
		}

		try (var stream = Files.list(featuresDir)) {
			return stream.filter(Files::isDirectory)
					.map(p -> p.getFileName().toString()).sorted()
					.collect(Collectors.toList());
		}
	}
}