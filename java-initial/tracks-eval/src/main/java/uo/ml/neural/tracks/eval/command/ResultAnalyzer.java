package uo.ml.neural.tracks.eval.command;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import uo.ml.neural.tracks.eval.model.BaselineResult;
import uo.ml.neural.tracks.eval.model.LofoResult;

/**
 * Analyzes LOFO validation and baseline results from various sources.
 * Extracts performance metrics and organizes them for statistical analysis.
 */
public class ResultAnalyzer {
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    // Pattern to extract metrics from LOFO output text
    private static final Pattern LOFO_METRICS_PATTERN = Pattern.compile(
        "(\\w+)\\s*\\t\\s*(\\d+\\.\\d+)\\s*\\t\\s*\\t\\s*(\\d+\\.\\d+)"
    );
    
    // Pattern to extract overall statistics
    private static final Pattern STATS_PATTERN = Pattern.compile(
        "(Distance|Elev \\(\\+\\)|Elev \\(\\-\\)|Overall):\\s*(\\d+\\.\\d+)\\s*±\\s*(\\d+\\.\\d+)"
    );
    
    /**
     * Analyzes results from the specified directory.
     * Looks for LOFO output files, JSON results, and baseline comparisons.
     */
    public EvaluationResults analyzeResults(Path resultsDir) throws IOException {
        List<LofoResult> lofoResults = new ArrayList<>();
        List<BaselineResult> baselineResults = new ArrayList<>();
        
        // Look for different types of result files
        if (Files.isDirectory(resultsDir)) {
            try (var stream = Files.list(resultsDir)) {
                List<Path> files = stream.toList();
                
                for (Path file : files) {
                    if (file.getFileName().toString().toLowerCase().contains("lofo")) {
                        parseLofoResults(file, lofoResults);
                    } else if (file.getFileName().toString().toLowerCase().contains("baseline")) {
                        parseBaselineResults(file, baselineResults);
                    } else if (file.getFileName().toString().endsWith(".json")) {
                        parseJsonResults(file, lofoResults, baselineResults);
                    } else if (file.getFileName().toString().endsWith(".txt") || 
                              file.getFileName().toString().endsWith(".log")) {
                        parseTextResults(file, lofoResults, baselineResults);
                    }
                }
            }
        } else {
            // Single file analysis
            if (resultsDir.getFileName().toString().endsWith(".json")) {
                parseJsonResults(resultsDir, lofoResults, baselineResults);
            } else {
                parseTextResults(resultsDir, lofoResults, baselineResults);
            }
        }
        
        return new EvaluationResults(lofoResults, baselineResults);
    }
    
    private void parseLofoResults(Path file, List<LofoResult> results) throws IOException {
        String content = Files.readString(file);
        
        // Look for individual fold results
        String[] lines = content.split("\n");
        Map<String, Double> nnResults = new HashMap<>();
        Map<String, Double> baselineResults = new HashMap<>();
        
        boolean inResultsSection = false;
        
        for (String line : lines) {
            line = line.trim();
            
            if (line.contains("Results by family:") || line.contains("Family\t")) {
                inResultsSection = true;
                continue;
            }
            
            if (inResultsSection && line.contains("\t")) {
                Matcher matcher = LOFO_METRICS_PATTERN.matcher(line);
                if (matcher.find()) {
                    String family = matcher.group(1);
                    double nnMAE = Double.parseDouble(matcher.group(2));
                    double baseMAE = Double.parseDouble(matcher.group(3));
                    
                    results.add(new LofoResult(family, nnMAE, baseMAE, 
                        new double[]{nnMAE, 0, 0}, new double[]{baseMAE, 0, 0}));
                }
            }
            
            // Parse overall statistics
            Matcher statsMatcher = STATS_PATTERN.matcher(line);
            if (statsMatcher.find()) {
                String metric = statsMatcher.group(1);
                double mean = Double.parseDouble(statsMatcher.group(2));
                double std = Double.parseDouble(statsMatcher.group(3));
                
                if (line.toLowerCase().contains("neural network")) {
                    nnResults.put(metric, mean);
                } else if (line.toLowerCase().contains("baseline")) {
                    baselineResults.put(metric, mean);
                }
            }
        }
    }
    
    private void parseBaselineResults(Path file, List<BaselineResult> results) throws IOException {
        String content = Files.readString(file);
        
        // Parse baseline-specific results
        String[] lines = content.split("\n");
        for (String line : lines) {
            // Look for baseline comparison data
            if (line.contains("MAE") && line.contains(":")) {
                String[] parts = line.split(":");
                if (parts.length >= 2) {
                    try {
                        String[] values = parts[1].trim().split("\\s+");
                        if (values.length >= 3) {
                            double dist = Double.parseDouble(values[0].replaceAll("[\\[\\],]", ""));
                            double pos = Double.parseDouble(values[1].replaceAll("[\\[\\],]", ""));
                            double neg = Double.parseDouble(values[2].replaceAll("[\\[\\],]", ""));
                            
                            results.add(new BaselineResult("baseline", dist, pos, neg));
                        }
                    } catch (NumberFormatException e) {
                        // Skip malformed lines
                    }
                }
            }
        }
    }
    
    private void parseJsonResults(Path file, List<LofoResult> lofoResults, 
                                  List<BaselineResult> baselineResults) throws IOException {
        JsonNode root = objectMapper.readTree(file.toFile());
        
        // Parse structured JSON results
        if (root.has("lofo_results")) {
            JsonNode lofoNode = root.get("lofo_results");
            for (JsonNode fold : lofoNode) {
                String family = fold.get("family").asText();
                double nnMAE = fold.get("neural_network_mae").asDouble();
                double baseMAE = fold.get("baseline_mae").asDouble();
                
                JsonNode nnDetails = fold.get("neural_network_details");
                JsonNode baseDetails = fold.get("baseline_details");
                
                double[] nnMAEs = {
                    nnDetails.get("distance").asDouble(),
                    nnDetails.get("elevation_pos").asDouble(),
                    nnDetails.get("elevation_neg").asDouble()
                };
                
                double[] baseMAEs = {
                    baseDetails.get("distance").asDouble(),
                    baseDetails.get("elevation_pos").asDouble(),
                    baseDetails.get("elevation_neg").asDouble()
                };
                
                lofoResults.add(new LofoResult(family, nnMAE, baseMAE, nnMAEs, baseMAEs));
            }
        }
    }
    
    private void parseTextResults(Path file, List<LofoResult> lofoResults, 
                                  List<BaselineResult> baselineResults) throws IOException {
        String content = Files.readString(file);
        
        // Try to parse as LOFO results first
        parseLofoResults(file, lofoResults);
        
        // If no LOFO results found, try baseline
        if (lofoResults.isEmpty()) {
            parseBaselineResults(file, baselineResults);
        }
    }
}