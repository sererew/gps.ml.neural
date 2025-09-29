package uo.ml.neural.tracks.preprocess.command;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import uo.ml.neural.tracks.core.geo.ProjectionUtils;
import uo.ml.neural.tracks.core.io.GpxUtils;
import uo.ml.neural.tracks.core.model.GpxPoint;
import uo.ml.neural.tracks.core.model.SegmentFeature;
import uo.ml.neural.tracks.core.model.UtmPoint;
import uo.ml.neural.tracks.core.preprocess.FeatureExtractor;
import uo.ml.neural.tracks.core.preprocess.Filters;
import uo.ml.neural.tracks.core.preprocess.Resampler3D;
import uo.ml.neural.tracks.preprocess.model.FilterType;
import uo.ml.neural.tracks.preprocess.model.TrackData;

public class TrackProcessor {
    private final Path gpxFile;
    private final FilterType filter;
    private final double stepMeters;

    public TrackProcessor(Path gpxFile, 
    		FilterType filter, 
    		double stepMeters) {
    	
        this.gpxFile = gpxFile;
        this.filter = filter;
        this.stepMeters = stepMeters;
    }

    public TrackData process() throws IOException {
        // Read GPX points
        List<GpxPoint> gpxPoints = GpxUtils.readGpx(gpxFile);
        if (gpxPoints.size() < 2) {
            throw new IOException("Track has fewer than 2 points: " + gpxFile);
        }
        
        // Convert to UTM
        ProjectionUtils.UtmZone utmZone = ProjectionUtils.detectZone(gpxPoints);
        List<UtmPoint> utmPoints = ProjectionUtils.toUtm(gpxPoints, utmZone);
        
        // Apply altitude filter if specified
        if (filter != FilterType.none) {
            utmPoints = applyAltitudeFilter(utmPoints);
        }
        
        // Resample by 3D arc length
        List<UtmPoint> resampledPoints = Resampler3D.resampleByArcLength3D(utmPoints, stepMeters);
        
        // Extract features
        List<SegmentFeature> features = FeatureExtractor.computeFeatures(resampledPoints);
        return new TrackData(features);
    }

    private List<UtmPoint> applyAltitudeFilter(List<UtmPoint> points) {
        List<Double> altitudes = points.stream()
        		.map(UtmPoint::getZ)
        		.toList();
        
        List<Double> filteredAltitudes;
        switch (filter) {
            case median:
                filteredAltitudes = Filters.medianFilter(altitudes, 5);
                break;
            case sgolay:
                filteredAltitudes = Filters.savitzkyGolay(altitudes, 5, 2);
                break;
            default:
                return points;
        }
        List<UtmPoint> filtered = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            UtmPoint original = points.get(i);
            filtered.add(new UtmPoint(original.getE(), original.getN(), filteredAltitudes.get(i)));
        }
        return filtered;
    }
}