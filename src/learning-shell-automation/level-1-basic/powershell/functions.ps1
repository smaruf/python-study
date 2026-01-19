# Level 1 - Basic: Functions in PowerShell

# Simple function
function Greet {
    Write-Host "Hello from a function!"
}

# Function with parameters
function Greet-User {
    param(
        [string]$Name,
        [string]$Role
    )
    Write-Host "Hello, $Name! Welcome to $Role learning."
}

# Function with return value
function Add-Numbers {
    param(
        [int]$Num1,
        [int]$Num2
    )
    return $Num1 + $Num2
}

# Function to check if service is running
function Check-Service {
    param(
        [string]$ServiceName
    )
    
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    
    if ($service -and $service.Status -eq 'Running') {
        Write-Host "Service $ServiceName is running" -ForegroundColor Green
        return $true
    } else {
        Write-Host "Service $ServiceName is not running" -ForegroundColor Yellow
        return $false
    }
}

# Advanced function with parameter validation
function Get-SystemUptime {
    [CmdletBinding()]
    param()
    
    $os = Get-CimInstance Win32_OperatingSystem
    $uptime = (Get-Date) - $os.LastBootUpTime
    
    Write-Host "System Uptime: $($uptime.Days) days, $($uptime.Hours) hours, $($uptime.Minutes) minutes"
}

# Call functions
Write-Host "=== Calling Functions ===" -ForegroundColor Cyan
Greet
Write-Host ""

Greet-User -Name "DevOps Engineer" -Role "Automation"
Write-Host ""

$result = Add-Numbers -Num1 10 -Num2 20
Write-Host "10 + 20 = $result"
Write-Host ""

# Check common Windows services
Write-Host "Checking services (example):"
Check-Service -ServiceName "Spooler"
Write-Host ""

# Show system uptime (Windows only)
if ($IsWindows -or $PSVersionTable.PSVersion.Major -lt 6) {
    Get-SystemUptime
}
