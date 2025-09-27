@echo off
REM Convenience script for running tracks-infer
REM Usage: infer.bat --model ./model/model.zip --scaler ./model/mu_sigma.json --gpx ./track.gpx [--step 1.0] [--filter none]

cd /d "%~dp0"
java -cp "tracks-infer-1.0.0-SNAPSHOT.jar;lib/*" uo.ml.neural.tracks.infer.InferCli %*