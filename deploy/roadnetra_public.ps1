# RoadNetra public demo: runs the app on this laptop behind a free Cloudflare quick tunnel.
#
#   powershell -ExecutionPolicy Bypass -File deploy\roadnetra_public.ps1 start    # start in the background
#   powershell -ExecutionPolicy Bypass -File deploy\roadnetra_public.ps1 status   # show the public link
#   powershell -ExecutionPolicy Bypass -File deploy\roadnetra_public.ps1 stop     # stop everything
#
# A hidden supervisor keeps the laptop awake, restarts the server if it crashes and the tunnel if it
# drops, and writes the current link to deploy\PUBLIC_URL.txt (the quick-tunnel link changes on restart).
param([ValidateSet("start", "stop", "status", "supervise")][string]$Action = "start")

$root = Split-Path -Parent $PSScriptRoot
$app = Join-Path $root "FINAL PROJECT"
$urlFile = Join-Path $PSScriptRoot "PUBLIC_URL.txt"
$pidFile = Join-Path $env:TEMP "roadnetra_supervisor.pid"
$logDir = Join-Path $env:TEMP "roadnetra"
$port = 8080
New-Item -ItemType Directory -Force $logDir | Out-Null

function Get-Python {
    $venv = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv } else { return (Get-Command python).Source }
}
function Get-Cloudflared {
    $c = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
    if (-not $c) { $c = "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe" }
    if (-not (Test-Path $c)) { throw "cloudflared not found. Install it: winget install --id Cloudflare.cloudflared -e" }
    return $c
}
function Get-Supervisor {
    if (-not (Test-Path $pidFile)) { return $null }
    $p = Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -like "powershell*") { return $p } else { return $null }
}
function Test-Server { try { Invoke-RestMethod "http://localhost:$port/api/health" -TimeoutSec 3 | Out-Null; $true } catch { $false } }
function Test-Url($u) { try { (Invoke-WebRequest "$u/api/health" -UseBasicParsing -TimeoutSec 10).StatusCode -eq 200 } catch { $false } }
function Show-Links {
    $u = if (Test-Path $urlFile) { (Get-Content $urlFile -Raw).Trim() } else { "" }
    if (-not $u) { Write-Host "No public link yet (the tunnel is starting)."; return }
    Write-Host "RoadNetra is live:" -ForegroundColor Green
    "  Command center : $u/", "  PWD portal     : $u/pwd", "  Hospital portal: $u/hospital",
    "  Police portal  : $u/police", "  Dashboard      : $u/frontend/" | ForEach-Object { Write-Host $_ }
}

switch ($Action) {
    "start" {
        if (Get-Supervisor) { Write-Host "Already running."; Show-Links; break }
        Remove-Item $urlFile -ErrorAction SilentlyContinue
        $sup = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "supervise")
        $sup.Id | Set-Content $pidFile
        Write-Host "Starting RoadNetra (loading 3 models, opening the tunnel)..."
        for ($i = 0; $i -lt 90 -and -not (Test-Path $urlFile); $i++) { Start-Sleep 2 }
        Show-Links
    }
    "status" {
        if (Get-Supervisor) { Write-Host "Supervisor running. Server: $(if (Test-Server) {'up'} else {'down'})"; Show-Links }
        else { Write-Host "Not running. Start it with: roadnetra_public.ps1 start" }
    }
    "stop" {
        $sup = Get-Supervisor
        if ($sup) { Stop-Process -Id $sup.Id -Force }
        Get-CimInstance Win32_Process | Where-Object {
            ($_.Name -eq "cloudflared.exe" -and $_.CommandLine -like "*localhost:$port*") -or
            ($_.Name -eq "python.exe" -and $_.CommandLine -like "*server.py*" -and $_.CommandLine -notlike "*jedi*")
        } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Remove-Item $pidFile, $urlFile -ErrorAction SilentlyContinue
        Write-Host "RoadNetra public demo stopped."
    }
    "supervise" {
        # Keep the PC awake (not the display) while the demo is up; released when this process exits
        Add-Type -Namespace RN -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
        [RN.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        $python = Get-Python; $cf = Get-Cloudflared
        $server = $null; $tunnel = $null; $url = $null; $fails = 0
        while ($true) {
            if (-not (Test-Server) -and (-not $server -or $server.HasExited)) {
                $server = Start-Process -FilePath $python -ArgumentList "server.py" -WorkingDirectory $app -PassThru -WindowStyle Hidden `
                    -RedirectStandardOutput "$logDir\server.out.log" -RedirectStandardError "$logDir\server.err.log"
                for ($i = 0; $i -lt 90 -and -not (Test-Server); $i++) { Start-Sleep 2 }
            }
            if (-not $tunnel -or $tunnel.HasExited -or $fails -ge 3) {
                if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force }
                $tlog = "$logDir\tunnel.log"; Remove-Item $tlog -ErrorAction SilentlyContinue
                $tunnel = Start-Process -FilePath $cf -ArgumentList "tunnel --no-autoupdate --url http://localhost:$port" `
                    -PassThru -WindowStyle Hidden -RedirectStandardError $tlog
                $url = $null; $fails = 0
                for ($i = 0; $i -lt 45 -and -not $url; $i++) {
                    Start-Sleep 2
                    if (Test-Path $tlog) { $url = (Select-String -Path $tlog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1).Matches.Value }
                }
                if ($url) { for ($i = 0; $i -lt 15 -and -not (Test-Url $url); $i++) { Start-Sleep 2 }; $url | Set-Content $urlFile }
            }
            Start-Sleep 20
            if ($url -and (Test-Server)) { if (Test-Url $url) { $fails = 0 } else { $fails++ } }
        }
    }
}
