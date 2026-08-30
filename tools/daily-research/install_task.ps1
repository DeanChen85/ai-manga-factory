# install_task.ps1 — 通过 schtasks + XML 定义每日定时任务
$ErrorActionPreference = 'Continue'

$python = 'C:\Users\Dean\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$script = 'F:\new ai factory\tools\daily-research\daily_research.py'
$logDir = 'F:\new ai factory\tools\daily-research\logs'
if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$tasks = @(
    @{ Name = 'DSH-DailyResearch-09'; Time = '09:00' },
    @{ Name = 'DSH-DailyResearch-21'; Time = '21:00' }
)

# XML 路径转义：在 XML 里反斜杠必须是双反斜杠
$pyXml = $python.Replace('\', '\\')
$scXml = $script.Replace('\', '\\')
$wdXml = 'F:\\new ai factory\\tools\\daily-research'

foreach ($t in $tasks) {
    $logFile = Join-Path $logDir "$($t.Name).log"

    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>DSH daily research scan: GitHub + Reddit + HN for new AI video projects; commit &amp; push to DeanChen85/ai-manga-factory.</Description>
    <Author>Dean Chen</Author>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2099-01-01T$($t.Time):00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <ExecutionTimeLimit>PT15M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>$pyXml</Command>
      <Arguments>"$scXml"</Arguments>
      <WorkingDirectory>$wdXml</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

    $xmlPath = Join-Path $env:TEMP "$($t.Name).xml"
    [System.IO.File]::WriteAllText($xmlPath, $xml, [System.Text.Encoding]::Unicode)

    # 删除旧任务
    cmd.exe /c "schtasks /Delete /TN $($t.Name) /F" 2>&1 | Out-Null

    # 通过 XML 创建
    cmd.exe /c "schtasks /Create /TN $($t.Name) /XML `"$xmlPath`" /F" 2>&1 | Out-Null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  + created: $($t.Name) at $($t.Time) (log: $logFile)"
    } else {
        Write-Host "  ! FAILED: $($t.Name) (exit $LASTEXITCODE)"
        Write-Host "  xmlPath: $xmlPath"
        Write-Host "  first 200 chars:"
        Get-Content -LiteralPath $xmlPath -Encoding Unicode -TotalCount 1 | Select-Object -First 5
    }

    Remove-Item -LiteralPath $xmlPath -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== verify ==="
cmd.exe /c "schtasks /Query /FO LIST" 2>$null | Select-String -Pattern 'DSH-DailyResearch|TaskName|Next Run Time|Status:' | ForEach-Object { $_.Line }