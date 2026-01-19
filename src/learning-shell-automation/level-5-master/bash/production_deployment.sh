#!/bin/bash
# Level 5 - Master: Production-Ready Deployment System
# Complete deployment system with monitoring, rollback, and notifications

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/var/log/deployment}"
CONFIG_FILE="${CONFIG_FILE:-deployment.conf}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
EMAIL_RECIPIENTS="${EMAIL_RECIPIENTS:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
setup_logging() {
    mkdir -p "$LOG_DIR"
    DEPLOYMENT_LOG="$LOG_DIR/deployment_$(date +%Y%m%d_%H%M%S).log"
    exec 1> >(tee -a "$DEPLOYMENT_LOG")
    exec 2>&1
}

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}[WARN $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

# Metrics collection
METRICS_FILE="/tmp/deployment_metrics.json"

record_metric() {
    local metric_name=$1
    local metric_value=$2
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat >> "$METRICS_FILE" << EOF
{"timestamp":"$timestamp","metric":"$metric_name","value":$metric_value}
EOF
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    local checks_passed=true
    
    # Check disk space
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        error "Disk usage too high: ${disk_usage}%"
        checks_passed=false
    else
        log "✓ Disk usage: ${disk_usage}%"
    fi
    
    # Check memory
    local mem_available=$(free | awk 'NR==2 {print $7}')
    if [ "$mem_available" -lt 1000000 ]; then
        warn "Low memory available: ${mem_available}KB"
    else
        log "✓ Memory available: ${mem_available}KB"
    fi
    
    # Check required commands
    local required_commands=("kubectl" "docker" "git" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            warn "Command not found: $cmd (demo mode)"
        else
            log "✓ Command available: $cmd"
        fi
    done
    
    # Check network connectivity
    if ping -c 1 8.8.8.8 &> /dev/null; then
        log "✓ Network connectivity OK"
    else
        error "Network connectivity issues"
        checks_passed=false
    fi
    
    if [ "$checks_passed" = false ]; then
        error "Pre-deployment checks failed"
        return 1
    fi
    
    log "✓ All pre-deployment checks passed"
    return 0
}

# Backup current state
backup_current_state() {
    log "Backing up current deployment state..."
    
    local backup_dir="/tmp/deployment_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Simulate backing up current state
    echo "Current deployment state" > "$backup_dir/state.txt"
    echo "$backup_dir" > /tmp/last_backup_path.txt
    
    log "✓ State backed up to: $backup_dir"
    record_metric "backup_created" 1
}

# Deploy application
deploy_application() {
    local version=$1
    local environment=$2
    
    log "Deploying application version $version to $environment..."
    
    local start_time=$(date +%s)
    
    # Deployment steps
    log "1. Pulling image..."
    sleep 1
    
    log "2. Updating deployment..."
    sleep 1
    
    log "3. Waiting for pods to be ready..."
    sleep 2
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "✓ Deployment completed in ${duration}s"
    record_metric "deployment_duration_seconds" "$duration"
    
    return 0
}

# Health check with retry
health_check() {
    local endpoint=$1
    local max_attempts=${2:-30}
    local retry_delay=${3:-10}
    
    log "Running health checks on $endpoint..."
    
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "${endpoint}/health" > /dev/null 2>&1; then
            log "✓ Health check passed (attempt $attempt)"
            record_metric "health_check_attempts" "$attempt"
            return 0
        fi
        
        warn "Health check failed (attempt $attempt/$max_attempts)"
        sleep "$retry_delay"
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
    record_metric "health_check_failed" 1
    return 1
}

# Smoke tests
run_smoke_tests() {
    log "Running smoke tests..."
    
    local tests_passed=0
    local tests_failed=0
    
    # Test 1: Basic connectivity
    log "Test 1: Basic connectivity"
    if curl -sf "http://localhost:8080" > /dev/null 2>&1; then
        log "  ✓ Passed"
        ((tests_passed++))
    else
        error "  ✗ Failed"
        ((tests_failed++))
    fi
    
    # Test 2: API endpoint
    log "Test 2: API endpoint"
    sleep 1
    log "  ✓ Passed (simulated)"
    ((tests_passed++))
    
    # Test 3: Database connectivity
    log "Test 3: Database connectivity"
    sleep 1
    log "  ✓ Passed (simulated)"
    ((tests_passed++))
    
    log "Smoke tests: $tests_passed passed, $tests_failed failed"
    record_metric "smoke_tests_passed" "$tests_passed"
    record_metric "smoke_tests_failed" "$tests_failed"
    
    [ $tests_failed -eq 0 ]
}

# Rollback
rollback() {
    error "Initiating rollback..."
    
    local backup_path=$(cat /tmp/last_backup_path.txt 2>/dev/null || echo "/tmp/no_backup")
    
    log "Restoring from backup: $backup_path"
    sleep 2
    
    log "✓ Rollback completed"
    record_metric "rollback_executed" 1
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    log "Sending notification: $status - $message"
    
    # Slack notification (simulated)
    if [ -n "$SLACK_WEBHOOK" ]; then
        log "Sending to Slack..."
    fi
    
    # Email notification (simulated)
    if [ -n "$EMAIL_RECIPIENTS" ]; then
        log "Sending email to $EMAIL_RECIPIENTS..."
    fi
    
    record_metric "notification_sent" 1
}

# Main deployment workflow
main() {
    local version=${1:-latest}
    local environment=${2:-production}
    
    setup_logging
    
    log "========================================="
    log "Production Deployment System"
    log "Version: $version"
    log "Environment: $environment"
    log "========================================="
    
    # Pre-deployment checks
    if ! pre_deployment_checks; then
        send_notification "FAILED" "Pre-deployment checks failed"
        exit 1
    fi
    
    # Backup current state
    backup_current_state
    
    # Deploy
    if ! deploy_application "$version" "$environment"; then
        error "Deployment failed"
        rollback
        send_notification "FAILED" "Deployment failed - rolled back"
        exit 1
    fi
    
    # Health check
    if ! health_check "http://localhost:8080" 10 5; then
        error "Health check failed"
        rollback
        send_notification "FAILED" "Health check failed - rolled back"
        exit 1
    fi
    
    # Smoke tests
    if ! run_smoke_tests; then
        error "Smoke tests failed"
        rollback
        send_notification "FAILED" "Smoke tests failed - rolled back"
        exit 1
    fi
    
    # Success
    log "========================================="
    log "✓ Deployment completed successfully!"
    log "========================================="
    
    send_notification "SUCCESS" "Deployment completed successfully"
    
    # Show metrics
    log "Deployment metrics saved to: $METRICS_FILE"
    
    return 0
}

# Execute
main "$@"
