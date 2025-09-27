@echo off
REM Convenience script for running tracks-train final training
REM Usage: final-train.bat --data ./data/processed --out ./model [--epochs 150] [--lr 0.001]

cd /d "%~dp0"
java -cp "tracks-train-1.0.0-SNAPSHOT.jar;lib/*" uo.ml.neural.tracks.train.FinalTrainCli %*