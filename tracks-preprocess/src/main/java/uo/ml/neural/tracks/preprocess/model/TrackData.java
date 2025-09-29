package uo.ml.neural.tracks.preprocess.model;

import java.util.List;

import uo.ml.neural.tracks.core.model.SegmentFeature;

public class TrackData {
    public final List<SegmentFeature> features;
    
    public TrackData(List<SegmentFeature> features) {
        this.features = features;
    }
}
