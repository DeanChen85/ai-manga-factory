# install_task.ps1 — 一次性创建两个每日定时任务
$ErrorActionPreference = 'Stop'

$python = 'C:\Users\Dean\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$script = 'F:\new ai factory\tools\daily-research\daily_research.py'
$logDir = 'F:\new ai factory\tools\daily-research\logs'
if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$tasks = @(
    @{ Name = 'DSH-DailyResearch-09'; Time = '09:00' },
    @{ Name = 'DSH-DailyResearch-21'; Time = '21:00' }
)

foreach ($t in $tasks) {
    $logFile = Join-Path $logDir ("$($t.Name).log")
    $action = New-ScheduledTaskAction -Execute $python -ArgumentList "`"$($script)`"" -WorkingDirectory 'F:\new ai factory'
    $trigger = New-ScheduledTaskTrigger -Daily -At $t.Time
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest -LogonType S4U

    # Idempotent: remove if exists
    if (Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false
    }

    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Auto-scan GitHub/Reddit for new AI video projects; report to DeanChen85/ai-manga-factory" | Out-Null
    Write-Host "  + created: $($t.Name) at $($t.Time) (log: $logFile)"
}

Write-Host ""
Write-Host "Verify:"
Get-ScheduledTask | Where-Object { $_.TaskName -like 'DSH-DailyResearch*' } |
    Select-Object TaskName, State, @{n='NextRun';e={($_.NextRunTime)}} | Format-Table -AutoSize | Out-String