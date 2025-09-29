package uo.ml.neural.tracks.preprocess.command;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

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

	public Integer preprocess() throws Exception {
		System.out.println("GPS Tracks Preprocessing Tool");
		System.out.println("=============================");
		System.out.printf("Input directory: %s%n", inputDir);
		System.out.printf("Output directory: %s%n", outputDir);
		System.out.printf("Step size: %.1f meters%n", stepMeters);
		System.out.printf("Altitude filter: %s%n", filter);
		System.out.println();

		// Validate input directory
		if (!Files.exists(inputDir) || !Files.isDirectory(inputDir)) {
			System.err.println(
					"Error: Input directory does not exist or is not a directory: "
							+ inputDir);
			return 1;
		}

		// Create output directory structure
		TracksIOUtils.createOutputDirectories(outputDir);

		// Find all families
		List<Path> familyDirs = findFamilyDirectories();
		if (familyDirs.isEmpty()) {
			System.err.println(
					"Error: No family directories found in input directory");
			return 1;
		}
		System.out.printf("Found %d families to process%n", familyDirs.size());

		// Process each family
		List<SegmentFeature> allFeatures = new ArrayList<>();
		for (Path familyDir : familyDirs) {
			String familyName = familyDir.getFileName().toString();
			System.out.printf("Processing family: %s%n", familyName);
			try {
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
						processedFamily.patternSteps);
				
			} catch (Exception e) {
				System.err.printf("Error processing family %s: %s%n",
						familyName, e.getMessage());
				e.printStackTrace();
				return 1;
			}
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
		return 0;
	}

	private List<Path> findFamilyDirectories() throws Exception {
		try (var stream = Files.list(inputDir)) {
			return stream.filter(Files::isDirectory)
					.collect(Collectors.toList());
		}
	}
}