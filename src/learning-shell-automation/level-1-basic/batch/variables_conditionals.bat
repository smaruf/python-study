@echo off
REM Level 1 - Basic: Variables and Conditionals in Batch

REM Variables
set name=DevOps Engineer
set age=25
set is_learning=true

REM Display variables
echo Role: %name%
echo Age: %age%

REM Conditional statements
if %age% GEQ 18 (
    echo You are an adult
) else (
    echo You are a minor
)

REM Check if learning
if "%is_learning%"=="true" (
    echo Keep learning and growing!
)

REM String comparison
set environment=production
if "%environment%"=="production" (
    echo Running in production mode
) else if "%environment%"=="development" (
    echo Running in development mode
) else (
    echo Unknown environment
)

pause
