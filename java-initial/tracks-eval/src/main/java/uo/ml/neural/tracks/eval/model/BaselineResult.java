package uo.ml.neural.tracks.eval.model;

/**
 * Represents baseline results for comparison analysis.
 */
public class BaselineResult {
    
    private final String name;
    private final double distanceMAE;
    private final double elevationPosMAE;
    private final double elevationNegMAE;
    
    public BaselineResult(String name, double distanceMAE, double elevationPosMAE, double elevationNegMAE) {
        this.name = name;
        this.distanceMAE = distanceMAE;
        this.elevationPosMAE = elevationPosMAE;
        this.elevationNegMAE = elevationNegMAE;
    }
    
    public String getName() {
        return name;
    }
    
    public double getDistanceMAE() {
        return distanceMAE;
    }
    
    public double getElevationPosMAE() {
        return elevationPosMAE;
    }
    
    public double getElevationNegMAE() {
        return elevationNegMAE;
    }
    
    public double getOverallMAE() {
        return (distanceMAE + elevationPosMAE + elevationNegMAE) / 3.0;
    }
    
    @Override
    public String toString() {
        return String.format("BaselineResult{name='%s', distance=%.3f, elev_pos=%.3f, elev_neg=%.3f, overall=%.3f}",
            name, distanceMAE, elevationPosMAE, elevationNegMAE, getOverallMAE());
    }
}