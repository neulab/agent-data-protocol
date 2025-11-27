# PowerShell script to convert all 9 dataset samples: Raw -> Standardized -> SFT (OpenHands format)
# Run this from the agent-data-collection root directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Dataset Sample Conversion Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set PYTHONPATH to current directory
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Array of datasets with their parameters
$datasets = @(
    @{Name="android_in_the_wild"; IsWeb="no"; ApiEnv="execute_bash"},
    @{Name="androidcontrol"; IsWeb="no"; ApiEnv="execute_bash"},
    @{Name="llava_plus"; IsWeb="no"; ApiEnv="execute_bash"},
    @{Name="omniact"; IsWeb="no"; ApiEnv="execute_bash"},
    @{Name="webarena_successful"; IsWeb="yes"; ApiEnv="browser"},
    @{Name="weblinx"; IsWeb="yes"; ApiEnv="browser"},
    @{Name="wonderbread"; IsWeb="yes"; ApiEnv="browser"},
    @{Name="go-browse-wa"; IsWeb="yes"; ApiEnv="browser"},
    @{Name="openhands"; IsWeb="no"; ApiEnv="execute_bash"}
)

$count = 0
foreach ($dataset in $datasets) {
    $count++
    $name = $dataset.Name
    $isWeb = $dataset.IsWeb
    $apiEnv = $dataset.ApiEnv

    Write-Host "[$count/9] Processing $name..." -ForegroundColor Yellow

    $rawPath = "datasets\$name\sample_raw.json"
    $stdPath = "datasets\$name\sample_std.json"
    $sftPath = "datasets\$name\sample_sft.json"

    if (Test-Path $rawPath) {
        Write-Host "  - Converting raw to standardized..." -ForegroundColor Gray

        # Raw -> Standardized
        Get-Content $rawPath |
            python scripts\json_to_jsonl.py |
            python datasets\$name\raw_to_standardized.py |
            python scripts\jsonl_to_json.py |
            Set-Content $stdPath

        Write-Host "  - Converting standardized to SFT (OpenHands)..." -ForegroundColor Gray

        # Set MY_DATASET environment variable for this dataset
        $env:MY_DATASET = $name

        # Standardized -> SFT
        Get-Content $stdPath |
            python scripts\json_to_jsonl.py |
            python agents\openhands\std_to_sft.py --is_web=$isWeb --api_env=$apiEnv |
            python scripts\jsonl_to_json.py |
            Set-Content $sftPath

        Write-Host "  - Done!" -ForegroundColor Green
    } else {
        Write-Host "  - SKIPPED: sample_raw.json not found" -ForegroundColor DarkGray
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All datasets processed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results:" -ForegroundColor White
Write-Host "- sample_std.json files have been regenerated (with ImageObservation schema fixes)" -ForegroundColor White
Write-Host "- sample_sft.json files contain OpenHands format conversions" -ForegroundColor White
Write-Host ""
