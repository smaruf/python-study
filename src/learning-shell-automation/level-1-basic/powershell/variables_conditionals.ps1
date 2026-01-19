# Level 1 - Basic: Variables and Conditionals in PowerShell

# Variables
$name = "DevOps Engineer"
$age = 25
$isLearning = $true

# Display variables
Write-Host "Role: $name"
Write-Host "Age: $age"

# Conditional statements
if ($age -ge 18) {
    Write-Host "You are an adult"
} else {
    Write-Host "You are a minor"
}

# Check if learning
if ($isLearning) {
    Write-Host "Keep learning and growing!"
}

# Multiple conditions
if (($age -ge 18) -and $isLearning) {
    Write-Host "Perfect! Adult learner on the path to DevOps mastery!"
}

# Switch statement
$day = "Monday"
switch ($day) {
    "Monday" { Write-Host "Start of the work week!" }
    "Friday" { Write-Host "Almost weekend!" }
    "Saturday" { Write-Host "Weekend is here!" }
    "Sunday" { Write-Host "Weekend is here!" }
    default { Write-Host "It's a regular day" }
}
