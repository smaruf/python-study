# Level 0 - Beginner: Basic System Information in PowerShell

Write-Host "=== System Information ==="
Write-Host "Hostname: $env:COMPUTERNAME"
Write-Host "Current User: $env:USERNAME"
Write-Host "Current Directory: $(Get-Location)"
Write-Host "Date and Time: $(Get-Date)"
Write-Host "PowerShell Version: $($PSVersionTable.PSVersion)"
Write-Host "=========================="
