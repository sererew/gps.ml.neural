param(
    [ValidateSet("init-data", "run", "fetch", "all")]
    [string]$Action = "run",

    [string]$Remote = "miguel@156.35.160.77",
    [string]$RemoteDir = "/home/miguel/alb/gps.ml.neural",
    [int]$Seed = 42,
    [string]$Python = "python3"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$LocalDataInput = Join-Path $RepoRoot "data\input"
$LocalTrainScript = Join-Path $RepoRoot "python\pipeline\6_train_neural_network_v3.py"
$LocalRequirements = Join-Path $RepoRoot "python\requirements.txt"
$LocalResultsDir = Join-Path $RepoRoot "results\training\a40"
$LocalModelsDir = Join-Path $RepoRoot "models\a40"

function Invoke-Remote {
    param([string]$Command)
    ssh $Remote $Command
}

function Copy-ToRemote {
    param(
        [string]$Source,
        [string]$Target
    )
    scp -r $Source "${Remote}:${Target}"
}

function Copy-FromRemote {
    param(
        [string]$Source,
        [string]$Target
    )
    scp -r "${Remote}:${Source}" $Target
}

function Initialize-RemoteLayout {
    Invoke-Remote "mkdir -p '$RemoteDir/data' '$RemoteDir/python/pipeline' '$RemoteDir/python' '$RemoteDir/results/training' '$RemoteDir/models'"
}

function Initialize-Data {
    if (-not (Test-Path $LocalDataInput)) {
        throw "Local data folder not found: $LocalDataInput"
    }

    Initialize-RemoteLayout
    Write-Host "Copying data/input to ${Remote}:$RemoteDir/data/"
    Copy-ToRemote $LocalDataInput "$RemoteDir/data/"
}

function Copy-TrainingFiles {
    if (-not (Test-Path $LocalTrainScript)) {
        throw "Training script not found: $LocalTrainScript"
    }
    if (-not (Test-Path $LocalRequirements)) {
        throw "Requirements file not found: $LocalRequirements"
    }

    Initialize-RemoteLayout
    Write-Host "Copying training script and requirements"
    Copy-ToRemote $LocalTrainScript "$RemoteDir/python/pipeline/6_train_neural_network_v3.py"
    Copy-ToRemote $LocalRequirements "$RemoteDir/python/requirements.txt"
}

function Invoke-RemoteTraining {
    Copy-TrainingFiles

    $remoteCommand = @"
cd '$RemoteDir' &&
if [ ! -d venv ]; then $Python -m venv venv; fi &&
. venv/bin/activate &&
pip install -r python/requirements.txt &&
python -X utf8 python/pipeline/6_train_neural_network_v3.py --seed $Seed 2>&1 | tee results/training/training_v3_complete_a40.log
"@

    Write-Host "Running full training on $Remote"
    Invoke-Remote $remoteCommand
}

function Fetch-RemoteResults {
    New-Item -ItemType Directory -Force -Path $LocalResultsDir | Out-Null
    New-Item -ItemType Directory -Force -Path $LocalModelsDir | Out-Null

    Write-Host "Fetching training results"
    Copy-FromRemote "$RemoteDir/results/training/training_results_v3_complete.json" (Join-Path $LocalResultsDir "training_results_v3_complete.json")
    Copy-FromRemote "$RemoteDir/results/training/training_history_v3.png" (Join-Path $LocalResultsDir "training_history_v3.png")
    Copy-FromRemote "$RemoteDir/results/training/training_v3_complete_a40.log" (Join-Path $LocalResultsDir "training_v3_complete_a40.log")

    Write-Host "Fetching models"
    Copy-FromRemote "$RemoteDir/models/model_best_v3.keras" (Join-Path $LocalModelsDir "model_best_v3.keras")
    Copy-FromRemote "$RemoteDir/models/model_final_v3.keras" (Join-Path $LocalModelsDir "model_final_v3.keras")
}

switch ($Action) {
    "init-data" {
        Initialize-Data
    }
    "run" {
        Invoke-RemoteTraining
    }
    "fetch" {
        Fetch-RemoteResults
    }
    "all" {
        Initialize-Data
        Invoke-RemoteTraining
        Fetch-RemoteResults
    }
}
