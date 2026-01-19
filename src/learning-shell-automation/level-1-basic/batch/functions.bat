@echo off
REM Level 1 - Basic: Functions (Subroutines) in Batch

echo === Calling Functions ===

REM Call a simple function
call :greet
echo.

REM Call function with parameters
call :greet_user "DevOps Engineer" "Automation"
echo.

REM Call function with return value
call :add_numbers 10 20
echo 10 + 20 = %result%
echo.

REM Call function to check directory
call :check_directory "C:\Windows"

pause
exit /b

REM === Function Definitions ===

:greet
echo Hello from a function!
exit /b

:greet_user
echo Hello, %~1! Welcome to %~2 learning.
exit /b

:add_numbers
set /a result=%~1 + %~2
exit /b

:check_directory
if exist %~1 (
    echo Directory %~1 exists
) else (
    echo Directory %~1 does not exist
)
exit /b
