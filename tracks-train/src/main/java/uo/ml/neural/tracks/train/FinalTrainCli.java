package uo.ml.neural.tracks.train;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import uo.ml.neural.tracks.core.exception.CommandException;
import uo.ml.neural.tracks.train.commands.finaltrain.FinalTrainingService;

/**
 * Final training CLI that trains a model on all available families
 * and saves the trained model and normalization parameters.
 */
@Command(
    name = "final-trainer",
    mixinStandardHelpOptions = true,
    version = "1.0.0-SNAPSHOT",
    description = "Trains final model on all families and saves for inference"
)
public class FinalTrainCli implements Callable<Integer> {
    @Option(names = {"--input"}, 
    		required = true, 
    		description = "Processed data directory (output of preprocessing)"
    )
    private Path dataDir;

    @Option(names = {"--output"}, 
    		required = true, 
    		description = "Output directory for trained model"
    )
    private Path outputDir;

    @Option(names = {"--epochs"}, 
    		defaultValue = "150", 
    		description = "Number of training epochs (default: ${DEFAULT-VALUE})"
    )
    private int epochs;

    @Option(names = {"--lr"}, 
    		defaultValue = "0.001", 
    		description = "Learning rate (default: ${DEFAULT-VALUE})"
    )
    private double learningRate;

    public static void main(String[] args) {
        int exitCode = new CommandLine(new FinalTrainCli()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() {
        try {
            new FinalTrainingService(dataDir, outputDir, epochs, learningRate).train();
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