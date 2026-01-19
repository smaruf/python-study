#!/bin/bash
# Level 2 - Intermediate: Process Management and Monitoring

# Function to check if process is running
check_process() {
    local process_name=$1
    
    if pgrep -x "$process_name" > /dev/null; then
        echo "✓ Process '$process_name' is running"
        return 0
    else
        echo "✗ Process '$process_name' is not running"
        return 1
    fi
}

# Function to get process info
get_process_info() {
    local process_name=$1
    
    echo "=== Process Information for $process_name ==="
    ps aux | grep -v grep | grep "$process_name" | head -5
}

# Function to monitor system resources
monitor_resources() {
    echo "=== System Resource Monitoring ==="
    
    # CPU usage
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "  Usage: " 100 - $1"%"}'
    
    # Memory usage
    echo "Memory Usage:"
    free -h | awk 'NR==2{printf "  Used: %s/%s (%.2f%%)\n", $3,$2,$3*100/$2 }'
    
    # Disk usage
    echo "Disk Usage:"
    df -h / | awk 'NR==2{printf "  Used: %s/%s (%s)\n", $3,$2,$5}'
}

# Function to get top processes by CPU
get_top_cpu_processes() {
    local count=${1:-5}
    
    echo "=== Top $count Processes by CPU Usage ==="
    ps aux --sort=-%cpu | head -n $((count + 1))
}

# Function to get top processes by Memory
get_top_mem_processes() {
    local count=${1:-5}
    
    echo "=== Top $count Processes by Memory Usage ==="
    ps aux --sort=-%mem | head -n $((count + 1))
}

# Main execution
echo "=== Process Management Demo ==="
echo ""

# Check for common processes
check_process "bash"
check_process "nonexistent_process"
echo ""

# Monitor resources
monitor_resources
echo ""

# Show top processes
get_top_cpu_processes 3
echo ""

get_top_mem_processes 3
