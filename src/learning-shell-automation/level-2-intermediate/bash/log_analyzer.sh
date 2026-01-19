#!/bin/bash
# Level 2 - Intermediate: Log File Analysis and Monitoring

LOG_FILE="./app.log"
ERROR_LOG="./errors.log"

# Create sample log file
create_sample_log() {
    echo "Creating sample log file..."
    cat > "$LOG_FILE" << EOF
2024-01-15 10:00:01 INFO Application started
2024-01-15 10:00:05 INFO User login: user123
2024-01-15 10:00:10 WARN Memory usage high: 85%
2024-01-15 10:00:15 INFO Processing request
2024-01-15 10:00:20 ERROR Database connection failed
2024-01-15 10:00:25 INFO Retrying connection
2024-01-15 10:00:30 INFO Connection successful
2024-01-15 10:00:35 ERROR File not found: config.yml
2024-01-15 10:00:40 WARN Disk space low: 90%
2024-01-15 10:00:45 INFO Request completed
EOF
    echo "✓ Sample log created"
}

# Function to count log levels
count_log_levels() {
    local log_file=$1
    
    echo "=== Log Level Summary ==="
    echo "INFO:  $(grep -c "INFO" "$log_file")"
    echo "WARN:  $(grep -c "WARN" "$log_file")"
    echo "ERROR: $(grep -c "ERROR" "$log_file")"
}

# Function to extract errors
extract_errors() {
    local log_file=$1
    local error_file=$2
    
    echo "Extracting errors to $error_file..."
    grep "ERROR" "$log_file" > "$error_file"
    echo "✓ Found $(wc -l < "$error_file") errors"
}

# Function to find recent errors
find_recent_errors() {
    local log_file=$1
    local minutes=${2:-60}
    
    echo "=== Recent Errors (last $minutes minutes) ==="
    # In real scenario, this would use actual timestamp comparison
    tail -n 20 "$log_file" | grep "ERROR"
}

# Function to monitor log in real-time (simulation)
monitor_log() {
    local log_file=$1
    
    echo "=== Monitoring Log File (showing last 5 lines) ==="
    tail -n 5 "$log_file"
    # In real usage: tail -f "$log_file"
}

# Function to analyze patterns
analyze_patterns() {
    local log_file=$1
    
    echo "=== Pattern Analysis ==="
    echo "Failed connections:"
    grep -i "failed\|error" "$log_file" | wc -l
    
    echo "Warning patterns:"
    grep -i "warn\|high\|low" "$log_file" | wc -l
}

# Main execution
create_sample_log
echo ""

count_log_levels "$LOG_FILE"
echo ""

extract_errors "$LOG_FILE" "$ERROR_LOG"
echo ""

find_recent_errors "$LOG_FILE"
echo ""

monitor_log "$LOG_FILE"
echo ""

analyze_patterns "$LOG_FILE"

echo ""
echo "Log analysis completed!"
