package uo.ml.neural.tracks.eval;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.Callable;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import uo.ml.neural.tracks.eval.command.EvaluationResults;
import uo.ml.neural.tracks.eval.command.ReportGenerator;
import uo.ml.neural.tracks.eval.command.ResultAnalyzer;

/**
 * CLI tool for evaluating LOFO and baseline results.
 * Generates statistical reports in CSV and Markdown formats.
 */
@Command(
    name = "tracks-eval",
    mixinStandardHelpOptions = true,
    version = "1.0.0-SNAPSHOT",
    description = "Analyzes LOFO validation and baseline results, generates statistical reports"
)
public class EvalCli implements Callable<Integer> {
    
    @Option(names = {"--results"}, required = true,
            description = "Directory containing LOFO results or result files")
    private Path resultsDir;
    
    @Option(names = {"--output"}, defaultValue = "./evaluation_report",
            description = "Output directory for reports (default: ${DEFAULT-VALUE})")
    private Path outputDir;
    
    @Option(names = {"--format"}, defaultValue = "both",
            description = "Output format: csv, markdown, or both (default: ${DEFAULT-VALUE})")
    private String format;
    
    public static void main(String[] args) {
        int exitCode = new CommandLine(new EvalCli()).execute(args);
        System.exit(exitCode);
    }
    
    @Override
    public Integer call() throws Exception {
        System.out.println("GPS Tracks Evaluation Tool");
        System.out.println("==========================");
        System.out.printf("Results directory: %s%n", resultsDir);
        System.out.printf("Output directory: %s%n", outputDir);
        System.out.printf("Format: %s%n", format);
        System.out.println();
        
        // Validate inputs
        if (!Files.exists(resultsDir)) {
            System.err.println("Error: Results directory does not exist: " + resultsDir);
            return 1;
        }
        
        // Create output directory
        Files.createDirectories(outputDir);
        
        try {
            // Load and analyze results
            System.out.println("Loading results...");
            ResultAnalyzer analyzer = new ResultAnalyzer();
            EvaluationResults results = analyzer.analyzeResults(resultsDir);
            
            System.out.printf("Found %d LOFO folds and %d baseline results%n", 
                results.getLofoResults().size(), results.getBaselineResults().size());
            
            // Generate reports
            System.out.println("Generating reports...");
            ReportGenerator generator = new ReportGenerator();
            
            if ("csv".equals(format) || "both".equals(format)) {
                generator.generateCSVReport(results, outputDir.resolve("evaluation_report.csv"));
                System.out.println("CSV report generated: " + outputDir.resolve("evaluation_report.csv"));
            }
            
            if ("markdown".equals(format) || "both".equals(format)) {
                generator.generateMarkdownReport(results, outputDir.resolve("evaluation_report.md"));
                System.out.println("Markdown report generated: " + outputDir.resolve("evaluation_report.md"));
            }
            
            // Generate summary statistics
            generator.generateSummaryReport(results, outputDir.resolve("summary.txt"));
            System.out.println("Summary report generated: " + outputDir.resolve("summary.txt"));
            
            System.out.println();
            System.out.println("Evaluation completed successfully!");
            return 0;
            
        } catch (Exception e) {
            System.err.println("Error during evaluation: " + e.getMessage());
            e.printStackTrace();
            return 1;
        }
    }
}