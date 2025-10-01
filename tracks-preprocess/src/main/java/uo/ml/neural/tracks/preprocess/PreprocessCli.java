package uo.ml.neural.tracks.preprocess;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.preprocess.command.TracksPreprocessingService;
import uo.ml.neural.tracks.preprocess.model.FilterType;

/**
 * CLI tool for preprocessing GPS tracks and generating training datasets.
 * Processes folders of GPX files and generates features, labels, and
 * normalization parameters.
 */
@Command(name = "tracks-preprocess", 
	mixinStandardHelpOptions = true, 
	version = "1.0.0-SNAPSHOT", 
	description = "Preprocesses GPS tracks from GPX files and generates ML training datasets"
)
public class PreprocessCli implements Callable<Integer> {
	@Option(names = {"--input" }, 
			required = true, 
			description = "Root directory containing family subdirectories with GPX files"
	)
	private Path inputDir;

	@Option(names = {"--output" }, 
			required = true, 
			description = "Output directory for processed datasets"
	)
	private Path outputDir;
	
	@Option(names = {"--step" }, 
			defaultValue = "1.0", 
			description = "Resampling step size in meters (default: ${DEFAULT-VALUE})"
	)
	private double stepMeters;
	
	@Option(names = {"--filter" }, 
			defaultValue = "none", 
			description = "Altitude filter: median, sgolay, or none (default: ${DEFAULT-VALUE})"
	)
	private FilterType filter;

	public static void main(String[] args) {
		int exitCode = new CommandLine(new PreprocessCli()).execute(args);
		System.exit(exitCode);
	}

	@Override
	public Integer call() {
		try {
			
			new TracksPreprocessingService(
					inputDir, 
					outputDir, 
					stepMeters, 
					filter
				).preprocess();
			
			return 0;
			
        } catch (CommandException e) {
            System.err.println("ERROR: " + e.getMessage());
            
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
        }

		return 1;
	}
}