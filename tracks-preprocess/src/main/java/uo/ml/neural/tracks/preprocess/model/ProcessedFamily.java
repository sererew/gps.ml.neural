package uo.ml.neural.tracks.preprocess.model;

import java.util.List;

import uo.ml.neural.tracks.core.model.SegmentFeature;

public class ProcessedFamily {
	public final int trackCount;
	public final int patternSteps;
	public final List<SegmentFeature> allNoisyFeatures;

	public ProcessedFamily(int trackCount, int patternSteps,
			List<SegmentFeature> allNoisyFeatures) {
		this.trackCount = trackCount;
		this.patternSteps = patternSteps;
		this.allNoisyFeatures = allNoisyFeatures;
	}
}
