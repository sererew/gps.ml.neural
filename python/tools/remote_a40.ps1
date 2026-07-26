param(
    [ValidateSet("init-data", "run", "fetch", "run-fetch", "all")]
    [string]$Action = "run",

    [string]$Remote = "gpu",
    [string]$RemoteDir = "/home/alb/gps.ml.neural",
    [int]$Seed = 42,
    [string]$Python = "python3.11",
    [string]$CudaVisibleDevices = "0",
    [string]$RemoteVenv = "venv_context_v1_py311",
    [ValidateSet("v3", "context_v1", "context_tcn_v1", "context_tcn_v2", "context_tcn_v3", "context_cascade_v1", "context_cascade_v2", "context_cascade_v3")]
    [string]$Training = "context_tcn_v2",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

foreach ($Arg in $RemainingArgs) {
    if ($Arg -like "-Action=*") {
        $Action = $Arg.Substring("-Action=".Length)
    }
    if ($Arg -like "-Training=*") {
        $Training = $Arg.Substring("-Training=".Length)
    }
}

if ($Action -notin @("init-data", "run", "fetch", "run-fetch", "all")) {
    throw "Invalid Action '$Action'. Use one of: init-data, run, fetch, run-fetch, all"
}

if ($Training -notin @("v3", "context_v1", "context_tcn_v1", "context_tcn_v2", "context_tcn_v3", "context_cascade_v1", "context_cascade_v2", "context_cascade_v3")) {
    throw "Invalid Training '$Training'. Use one of: v3, context_v1, context_tcn_v1, context_tcn_v2, context_tcn_v3, context_cascade_v1, context_cascade_v2, context_cascade_v3"
}

Write-Host "Remote action: $Action"
Write-Host "Remote training: $Training"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$DataInputName = if ($Training -eq "v3") { "input" } else { "input_context_v1" }
$RemoteDataRoot = "data/$DataInputName"
$LocalDataInput = Join-Path $RepoRoot "data\$DataInputName"
$TrainScriptName = "6_train_neural_network_$Training.py"
$LocalTrainScript = Join-Path $RepoRoot "python\pipeline\$TrainScriptName"
$LocalContextDatasetScript = Join-Path $RepoRoot "python\pipeline\5_generate_input_dataset_context_v1.py"
$LocalRequirements = Join-Path $RepoRoot "python\requirements-context-tf221-a40-lock.txt"
$LocalResultsDir = Join-Path $RepoRoot "results\training\a40"
$LocalModelsDir = Join-Path $RepoRoot "models\a40"
$LocalActiveModelsDir = Join-Path $RepoRoot "models"

function Invoke-Remote {
    param([string]$Command)
    $UnixCommand = $Command -replace "`r`n", "`n"
    $UnixCommand = $UnixCommand -replace "`r", ""
    $UnixCommand | ssh $Remote "tr -d '\r' | bash -s"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

function Copy-ToRemote {
    param(
        [string]$Source,
        [string]$Target
    )
    scp -r $Source "${Remote}:${Target}"
    if ($LASTEXITCODE -ne 0) {
        throw "Copy to remote failed with exit code $LASTEXITCODE"
    }
}

function Copy-FromRemote {
    param(
        [string]$Source,
        [string]$Target
    )
    scp -r "${Remote}:${Source}" $Target
    if ($LASTEXITCODE -ne 0) {
        throw "Copy from remote failed with exit code $LASTEXITCODE"
    }
}

function Copy-FromRemoteOptional {
    param(
        [string]$Source,
        [string]$Target
    )
    scp -r "${Remote}:${Source}" $Target
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Optional remote file not copied: $Source"
    }
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
    if (-not (Test-Path $LocalContextDatasetScript)) {
        throw "Context dataset script not found: $LocalContextDatasetScript"
    }

    Initialize-RemoteLayout
    Write-Host "Copying training script and requirements"
    Copy-ToRemote $LocalTrainScript "$RemoteDir/python/pipeline/$TrainScriptName"
    Copy-ToRemote $LocalContextDatasetScript "$RemoteDir/python/pipeline/5_generate_input_dataset_context_v1.py"
    Copy-ToRemote $LocalRequirements "$RemoteDir/python/requirements-context-tf221-a40-lock.txt"
}

function Invoke-RemoteTraining {
    Copy-TrainingFiles

    $remoteCommand = @"
set -euo pipefail
cd '$RemoteDir'
if [ ! -d '$RemoteVenv' ]; then $Python -m venv '$RemoteVenv'; fi
. '$RemoteVenv/bin/activate'
python --version
python -m pip install --upgrade pip
pip install -r python/requirements-context-tf221-a40-lock.txt
export CUDA_VISIBLE_DEVICES='$CudaVisibleDevices'
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus)
raise SystemExit(0 if gpus else 2)
PY
python -X utf8 python/pipeline/$TrainScriptName --data_root $RemoteDataRoot --seed $Seed 2>&1 | tee results/training/training_${Training}_complete_a40.log
"@

    Write-Host "Running $Training full training on $Remote using CUDA_VISIBLE_DEVICES=$CudaVisibleDevices"
    Invoke-Remote $remoteCommand
}

function Fetch-RemoteResults {
    New-Item -ItemType Directory -Force -Path $LocalResultsDir | Out-Null
    New-Item -ItemType Directory -Force -Path $LocalModelsDir | Out-Null

    Write-Host "Fetching training results"
    Copy-FromRemote "$RemoteDir/results/training/training_results_${Training}_complete.json" (Join-Path $LocalResultsDir "training_results_${Training}_complete.json")
    Copy-FromRemote "$RemoteDir/results/training/training_history_${Training}.png" (Join-Path $LocalResultsDir "training_history_${Training}.png")
    Copy-FromRemoteOptional "$RemoteDir/results/training/training_${Training}_complete_a40.log" (Join-Path $LocalResultsDir "training_${Training}_complete_a40.log")

    Write-Host "Fetching models"
    if ($Training -in @("context_cascade_v1", "context_cascade_v2", "context_cascade_v3")) {
        foreach ($Stage in @("fast", "slow")) {
            $ModelName = "model_final_${Training}_${Stage}"
            $LocalKerasModel = Join-Path $LocalModelsDir "$ModelName.keras"
            $LocalWeightsModel = Join-Path $LocalModelsDir "$ModelName.weights.h5"
            Copy-FromRemote "$RemoteDir/models/$ModelName.keras" $LocalKerasModel
            Copy-FromRemote "$RemoteDir/models/$ModelName.weights.h5" $LocalWeightsModel
            Copy-Item -Force -Path $LocalKerasModel -Destination (Join-Path $LocalActiveModelsDir "$ModelName.keras")
            Copy-Item -Force -Path $LocalWeightsModel -Destination (Join-Path $LocalActiveModelsDir "$ModelName.weights.h5")
        }
        Write-Host "Active local cascade models updated in $LocalActiveModelsDir"
        return
    }

    $LocalKerasModel = Join-Path $LocalModelsDir "model_final_${Training}.keras"
    $LocalWeightsModel = Join-Path $LocalModelsDir "model_final_${Training}.weights.h5"
    Copy-FromRemote "$RemoteDir/models/model_final_${Training}.keras" $LocalKerasModel
    Copy-FromRemote "$RemoteDir/models/model_final_${Training}.weights.h5" $LocalWeightsModel

    Copy-Item -Force -Path $LocalKerasModel -Destination (Join-Path $LocalActiveModelsDir "model_final_${Training}.keras")
    Copy-Item -Force -Path $LocalWeightsModel -Destination (Join-Path $LocalActiveModelsDir "model_final_${Training}.weights.h5")
    Write-Host "Active local model updated in $LocalActiveModelsDir"
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
    "run-fetch" {
        Invoke-RemoteTraining
        Fetch-RemoteResults
    }
    "all" {
        Initialize-Data
        Invoke-RemoteTraining
        Fetch-RemoteResults
    }
}
