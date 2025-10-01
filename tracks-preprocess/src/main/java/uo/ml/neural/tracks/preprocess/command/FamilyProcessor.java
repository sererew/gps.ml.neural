package uo.ml.neural.tracks.preprocess.command;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.core.exception.IO;
import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.preprocess.model.FilterType;
import uo.ml.neural.tracks.preprocess.model.ProcessedFamily;
import uo.ml.neural.tracks.preprocess.model.TrackData;

public class FamilyProcessor {
	private final Path familyDir;
	private final String familyName;
	private final Path outputDir;
	private final double stepMeters;
	private final FilterType filter;

	public FamilyProcessor(Path familyDir, String familyName, Path outputDir,
			double stepMeters, FilterType filter) {
		this.familyDir = familyDir;
		this.familyName = familyName;
		this.outputDir = outputDir;
		this.stepMeters = stepMeters;
		this.filter = filter;
	}

	public ProcessedFamily process() {
		List<Path> gpxFiles = findGpxFiles();
		Path patternFile = findPatternFile(gpxFiles);
		List<Path> noisyFiles = findNoisyFiles(gpxFiles, patternFile);
		System.out.printf("  Found %d noisy tracks and 1 pattern track%n",
				noisyFiles.size());
		
		TrackData patternTrack = processPatternTrack(patternFile);
		double[] labels = computeLabels(patternTrack.features);
		createFamilyOutputDirectories();
		saveLabels(labels);

		savePatternTrack(patternFile, patternTrack);
		List<SegmentFeature> noisyFeatures = processNoisyTracks(noisyFiles);
		
		return new ProcessedFamily(
				noisyFiles.size() + 1, // noisy files + the pattern 
				patternTrack.features.size(),
				noisyFeatures
			);
	}

	private List<Path> findGpxFiles() {
		return IO.get(() -> Files.list(familyDir)) 
				.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".gpx"))
				.toList();
	}

	private Path findPatternFile(List<Path> gpxFiles) {
		return gpxFiles.stream()
				.filter(p -> p.getFileName().toString().contains("_pattern.gpx"))
				.findFirst()
				.orElseThrow(() -> new CommandException(
					"No pattern file (*_pattern.gpx) found in family directory: "
							+ familyDir
						)
				);
	}

	private List<Path> findNoisyFiles(List<Path> gpxFiles, Path patternFile) {
		return gpxFiles.stream().filter(p -> !p.equals(patternFile)).toList();
	}

	private TrackData processPatternTrack(Path patternFile) {
		return new TrackProcessor(
				patternFile,
				filter, 
				stepMeters
			).process();
	}

	private void createFamilyOutputDirectories() {
		IO.exec( () ->Files.createDirectories(
				outputDir.resolve("features").resolve(familyName)
			));
	}

	private void saveLabels(double[] labels) {
		Path labelsFile = outputDir.resolve("labels").resolve(familyName + ".csv");
		TracksIOUtils.saveLabels(labelsFile, labels);
	}

	private void savePatternTrack(Path patternFile, TrackData patternTrack) {
		saveTrack(patternFile, patternTrack);
	}

	private void saveTrack(Path patternFile, TrackData patternTrack) {
		
		String patternBaseName = getBaseName(patternFile);
		
		Path csvFile = outputDir
					.resolve("features")
					.resolve(familyName)
					.resolve(patternBaseName + ".csv");
		
		TracksIOUtils.saveFeaturesCSV(csvFile, patternTrack.features);
	}

	private List<SegmentFeature> processNoisyTracks(List<Path> noisyFiles) {
		
		List<SegmentFeature> allNoisyFeatures = new ArrayList<>();
		for (Path noisyFile : noisyFiles) {
			TrackData trackData = new TrackProcessor(
					noisyFile,
					filter, 
					stepMeters
				).process();

			saveTrack(noisyFile, trackData);

			allNoisyFeatures.addAll(trackData.features);
		}
		return allNoisyFeatures;
	}

	private double[] computeLabels(List<SegmentFeature> features) {
		double distTotal = features.stream()
				.mapToDouble(SegmentFeature::getDh)
				.sum();
		
		double desnPos = features.stream()
				.mapToDouble(SegmentFeature::getDz)
				.filter(dz -> dz > 0)
				.sum();
		
		double desnNeg = Math.abs(features.stream()
					.mapToDouble(SegmentFeature::getDz)
					.filter(dz -> dz < 0)
					.sum()
				);
		
		return new double[] { distTotal, desnPos, desnNeg };
	}

	private String getBaseName(Path file) {
		String fileName = file.getFileName().toString();
		int dotIndex = fileName.lastIndexOf('.');
		return dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
	}
}