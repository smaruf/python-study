@echo off
REM Level 1 - Basic: Loops in Batch

echo === For Loop Example ===
REM Simple for loop
for %%i in (1 2 3 4 5) do (
    echo Number: %%i
)

echo.
echo === For Loop with Range ===
REM For loop with range
for /L %%i in (1,1,5) do (
    echo Count: %%i
)

echo.
echo === Loop through files ===
REM Loop through batch files
for %%f in (*.bat) do (
    echo Script found: %%f
)

echo.
echo === While Loop Simulation ===
REM Batch doesn't have while, but we can simulate with goto
set counter=1
:while_loop
if %counter% LEQ 5 (
    echo Counter: %counter%
    set /a counter+=1
    goto while_loop
)

pause
