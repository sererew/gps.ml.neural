package uo.ml.neural.tracks.preprocess.command;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import uo.ml.neural.tracks.core.exception.IO;
import uo.ml.neural.tracks.core.model.SegmentFeature;

public class TracksIOUtils {
	
	public static void saveFeaturesCSV(Path csvFile, List<SegmentFeature> features) {
		
		List<String> lines = new ArrayList<>();
		lines.add("dh,dz,slope"); // Header
		for (SegmentFeature feature : features) {
			lines.add(
				String.format(Locale.US, "%.6f,%.6f,%.6f", 
					feature.getDh(),
					feature.getDz(), 
					feature.getSlope()
				)
			);
		}
		IO.exec(() -> Files.write(csvFile, lines));
	}

	public static void saveLabels(Path labelsFile, double[] labels) {
		String content = String.format(Locale.US,
				"dist_total,desn_pos,desn_neg%n%.6f,%.6f,%.6f%n", 
				labels[0],
				labels[1], 
				labels[2]
			);
		IO.exec(() -> Files.writeString(labelsFile, content));
	}

	public static void createOutputDirectories(Path outputDir) {
		IO.exec(() -> {
			Files.createDirectories(outputDir);
			Files.createDirectories(outputDir.resolve("features"));
			Files.createDirectories(outputDir.resolve("labels"));
		});
	}
}