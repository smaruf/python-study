#!/bin/bash
# Level 3 - Advanced: CI/CD Pipeline Script
# This simulates a basic CI/CD pipeline workflow

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="my-app"
BUILD_DIR="./build"
DEPLOY_DIR="./deploy"
VERSION=$(date +"%Y%m%d.%H%M%S")

# Logging function
log() {
    local level=$1
    shift
    local message=$@
    
    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

# Stage 1: Code Checkout (simulated)
stage_checkout() {
    log INFO "Stage 1: Code Checkout"
    log INFO "Checking out code from repository..."
    sleep 1
    log INFO "✓ Code checkout completed"
}

# Stage 2: Build
stage_build() {
    log INFO "Stage 2: Build"
    
    mkdir -p "$BUILD_DIR"
    
    log INFO "Building application..."
    # Simulate build process
    cat > "$BUILD_DIR/${PROJECT_NAME}-${VERSION}.tar.gz" << EOF
This is a simulated build artifact
Version: $VERSION
Build Time: $(date)
EOF
    
    if [ $? -eq 0 ]; then
        log INFO "✓ Build completed successfully"
        log INFO "  Artifact: ${PROJECT_NAME}-${VERSION}.tar.gz"
        return 0
    else
        log ERROR "✗ Build failed"
        return 1
    fi
}

# Stage 3: Unit Tests
stage_test() {
    log INFO "Stage 3: Running Tests"
    
    log INFO "Running unit tests..."
    sleep 1
    
    # Simulate test results
    local passed=15
    local failed=0
    local total=$((passed + failed))
    
    log INFO "Test Results: $passed/$total passed"
    
    if [ $failed -eq 0 ]; then
        log INFO "✓ All tests passed"
        return 0
    else
        log ERROR "✗ $failed tests failed"
        return 1
    fi
}

# Stage 4: Code Quality Check
stage_quality() {
    log INFO "Stage 4: Code Quality Check"
    
    log INFO "Running linters..."
    sleep 1
    log INFO "✓ Linting passed"
    
    log INFO "Running security scan..."
    sleep 1
    log INFO "✓ No security vulnerabilities found"
    
    log INFO "Checking code coverage..."
    sleep 1
    log INFO "✓ Code coverage: 85%"
}

# Stage 5: Build Docker Image (simulated)
stage_docker() {
    log INFO "Stage 5: Building Docker Image"
    
    log INFO "Building Docker image: ${PROJECT_NAME}:${VERSION}"
    # Simulate docker build
    sleep 1
    
    log INFO "✓ Docker image built successfully"
    log INFO "  Image: ${PROJECT_NAME}:${VERSION}"
    log INFO "  Size: 125MB"
}

# Stage 6: Deploy
stage_deploy() {
    local environment=${1:-staging}
    
    log INFO "Stage 6: Deployment to $environment"
    
    mkdir -p "$DEPLOY_DIR/$environment"
    
    log INFO "Deploying to $environment environment..."
    cp "$BUILD_DIR/${PROJECT_NAME}-${VERSION}.tar.gz" "$DEPLOY_DIR/$environment/"
    
    log INFO "✓ Deployment completed"
    log INFO "  Environment: $environment"
    log INFO "  Version: $VERSION"
}

# Stage 7: Health Check
stage_health_check() {
    log INFO "Stage 7: Health Check"
    
    log INFO "Running health checks..."
    sleep 1
    
    log INFO "✓ Application is healthy"
    log INFO "  Status: Running"
    log INFO "  Response Time: 50ms"
}

# Main pipeline execution
main() {
    log INFO "========================================="
    log INFO "Starting CI/CD Pipeline"
    log INFO "Project: $PROJECT_NAME"
    log INFO "Version: $VERSION"
    log INFO "========================================="
    echo ""
    
    stage_checkout
    echo ""
    
    stage_build
    echo ""
    
    stage_test
    echo ""
    
    stage_quality
    echo ""
    
    stage_docker
    echo ""
    
    stage_deploy "staging"
    echo ""
    
    stage_health_check
    echo ""
    
    log INFO "========================================="
    log INFO "Pipeline completed successfully!"
    log INFO "========================================="
}

# Run the pipeline
main
