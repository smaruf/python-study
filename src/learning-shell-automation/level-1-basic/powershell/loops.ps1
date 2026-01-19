# Level 1 - Basic: Loops in PowerShell

Write-Host "=== ForEach Loop Example ==="
# ForEach loop
$numbers = 1..5
foreach ($num in $numbers) {
    Write-Host "Number: $num"
}

Write-Host "`n=== For Loop Example ==="
# For loop
for ($i = 1; $i -le 5; $i++) {
    Write-Host "Count: $i"
}

Write-Host "`n=== While Loop Example ==="
# While loop
$counter = 1
while ($counter -le 5) {
    Write-Host "Counter: $counter"
    $counter++
}

Write-Host "`n=== Loop through files ==="
# Loop through files
$files = Get-ChildItem -Filter "*.ps1"
foreach ($file in $files) {
    Write-Host "Script found: $($file.Name)"
}

Write-Host "`n=== Do-While Loop ==="
# Do-While loop
$num = 1
do {
    Write-Host "Do-While iteration: $num"
    $num++
} while ($num -le 3)
