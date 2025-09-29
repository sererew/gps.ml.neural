package uo.ml.neural.tracks.train;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.train.commands.lofo.LofoTrainingService;

/**
 * Leave-One-Family-Out (LOFO) cross-validation trainer for GPS track analysis.
 * Trains models excluding one family at a time and evaluates on the excluded
 * family.
 */
@Command(name = "lofo-trainer", 
		mixinStandardHelpOptions = true, 
		version = "1.0.0-SNAPSHOT", 
		description = "Performs Leave-One-Family-Out cross-validation training"
)
public class LofoTrainerCli implements Callable<Integer> {

	@Option(names = {"--input" }, 
			required = true, 
			description = "Processed data directory (output of preprocessing)"
	)
	private Path dataDir;

	@Option(names = {"--epochs" }, 
			defaultValue = "100", 
			description = "Maximum number of training epochs (default: ${DEFAULT-VALUE})"
	)
	private int maxEpochs;

	@Option(names = {"--lr" }, 
			defaultValue = "0.001", 
			description = "Learning rate (default: ${DEFAULT-VALUE})"
	)
	private double learningRate;

	public static void main(String[] args) {
		int exitCode = new CommandLine(new LofoTrainerCli()).execute(args);
		System.exit(exitCode);
	}

	@Override
	public Integer call() throws Exception {
		try {
			
			new LofoTrainingService(
						dataDir, 
						maxEpochs, 
						learningRate
					)
					.performLofoTraining();

			return 0;

		} catch (CommandException e) {
            System.err.println("ERROR: " + e.getMessage());
            return 1;
            
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            return 1;
        }
	}
}