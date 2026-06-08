@echo off
REM Convenience script for running tracks-train LOFO validation
REM Usage: lofo-train.bat --data ./data/processed [--epochs 100] [--lr 0.001]

cd /d "%~dp0"
java -cp "tracks-train-1.0.0-SNAPSHOT.jar;lib/*" uo.ml.neural.tracks.train.LofoTrainerCli %*