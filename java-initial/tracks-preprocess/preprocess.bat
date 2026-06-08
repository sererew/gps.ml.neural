@echo off
REM Convenience script for running tracks-preprocess
REM Usage: preprocess.bat --input ./input --output ./output [--step 1.0] [--filter none]

cd /d "%~dp0"
java -cp "target/tracks-preprocess-1.0.0-SNAPSHOT.jar;lib/*" uo.ml.neural.tracks.preprocess.PreprocessCli %*