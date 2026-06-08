package uo.ml.neural.tracks.preprocess.command;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.core.preprocess.ZScoreScaler;
import uo.ml.neural.tracks.preprocess.model.FilterType;
import uo.ml.neural.tracks.preprocess.model.ProcessedFamily;

public class TracksPreprocessingService {
	private final Path inputDir;
	private final Path outputDir;
	private final double stepMeters;
	private final FilterType filter;

	public TracksPreprocessingService(Path inputDir, Path outputDir,
			double stepMeters, FilterType filter) {
		this.inputDir = inputDir;
		this.outputDir = outputDir;
		this.stepMeters = stepMeters;
		this.filter = filter;
	}

	public void preprocess() {
		printHeader();

		// Validate input directory
		if (!Files.exists(inputDir) || !Files.isDirectory(inputDir)) {
			throw new CommandException(
					"Input directory does not exist or is not a directory: "
							+ inputDir);
		}

		// Create output directory structure
		TracksIOUtils.createOutputDirectories(outputDir);

		// Find all families
		List<Path> familyDirs = findFamilyDirectories();
		if (familyDirs.isEmpty()) {
			throw new CommandException(
					"No family directories found in input directory");
		}
		System.out.printf("Found %d families to process%n", familyDirs.size());

		// Process each family
		List<SegmentFeature> allFeatures = new ArrayList<>();
		for (Path familyDir : familyDirs) {
			String familyName = familyDir.getFileName().toString();
			System.out.printf("Processing family: %s%n", familyName);

			ProcessedFamily processedFamily = 
					new FamilyProcessor(
						familyDir,
						familyName, 
						outputDir, 
						stepMeters, 
						filter
					).process();
			
			allFeatures.addAll(processedFamily.allNoisyFeatures);
			System.out.printf(
					"  Processed %d tracks, pattern track has %d steps%n",
					processedFamily.trackCount,
					processedFamily.patternSteps
				);
				
		}

		// Compute and save global Z-score scaler
		if (!allFeatures.isEmpty()) {
			ZScoreScaler scaler = ZScoreScaler.fit(allFeatures);
			Path scalerPath = outputDir.resolve("mu_sigma.json");
			scaler.save(scalerPath);
			System.out.printf("Saved global Z-score parameters to: %s%n",
					scalerPath);
			System.out.printf("Total features for normalization: %d%n",
					allFeatures.size());
		}

		System.out.println("Preprocessing completed successfully!");
	}

	private void printHeader() {
		System.out.println("GPS Tracks Preprocessing Tool");
		System.out.println("=============================");
		System.out.printf("Input directory: %s%n", inputDir);
		System.out.printf("Output directory: %s%n", outputDir);
		System.out.printf("Step size: %.1f meters%n", stepMeters);
		System.out.printf("Altitude filter: %s%n", filter);
		System.out.println();
	}

	private List<Path> findFamilyDirectories() {
		try (var stream = Files.list(inputDir)) {
			return stream
					.filter(Files::isDirectory)
					.toList();
		} catch (IOException e) {
			throw new CommandException(
					"Error reading input directory: " + e.getMessage(), e);
		}
	}
}