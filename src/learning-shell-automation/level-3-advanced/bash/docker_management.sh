#!/bin/bash
# Level 3 - Advanced: Docker Container Management

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
IMAGE_NAME="my-app"
CONTAINER_NAME="my-app-container"
PORT=8080

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        echo "This is a demonstration script. Docker commands shown below:"
        DOCKER_AVAILABLE=false
    else
        echo -e "${GREEN}✓ Docker is available${NC}"
        DOCKER_AVAILABLE=true
    fi
}

# Build Docker image
build_image() {
    echo -e "${GREEN}Building Docker image...${NC}"
    
    # Create a sample Dockerfile
    cat > Dockerfile << 'EOF'
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/
EXPOSE 80
EOF
    
    # Create sample index.html
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head><title>DevOps Demo</title></head>
<body>
    <h1>Hello from Docker!</h1>
    <p>This is a CI/CD demonstration</p>
</body>
</html>
EOF
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker build -t "$IMAGE_NAME:latest" .
        echo -e "${GREEN}✓ Image built: $IMAGE_NAME:latest${NC}"
    else
        echo "  docker build -t $IMAGE_NAME:latest ."
    fi
}

# List Docker images
list_images() {
    echo -e "${GREEN}Docker Images:${NC}"
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker images "$IMAGE_NAME"
    else
        echo "  docker images $IMAGE_NAME"
    fi
}

# Run container
run_container() {
    echo -e "${GREEN}Starting container...${NC}"
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        # Stop existing container if running
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        
        # Run new container
        docker run -d \
            --name "$CONTAINER_NAME" \
            -p "$PORT:80" \
            "$IMAGE_NAME:latest"
        
        echo -e "${GREEN}✓ Container started${NC}"
        echo "  Access at: http://localhost:$PORT"
    else
        echo "  docker run -d --name $CONTAINER_NAME -p $PORT:80 $IMAGE_NAME:latest"
    fi
}

# List running containers
list_containers() {
    echo -e "${GREEN}Running Containers:${NC}"
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker ps -a | grep "$CONTAINER_NAME"
    else
        echo "  docker ps -a | grep $CONTAINER_NAME"
    fi
}

# View container logs
view_logs() {
    echo -e "${GREEN}Container Logs:${NC}"
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker logs "$CONTAINER_NAME" --tail 20
    else
        echo "  docker logs $CONTAINER_NAME --tail 20"
    fi
}

# Stop container
stop_container() {
    echo -e "${YELLOW}Stopping container...${NC}"
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker stop "$CONTAINER_NAME"
        echo -e "${GREEN}✓ Container stopped${NC}"
    else
        echo "  docker stop $CONTAINER_NAME"
    fi
}

# Clean up
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        docker rmi "$IMAGE_NAME:latest" 2>/dev/null || true
        rm -f Dockerfile index.html
        echo -e "${GREEN}✓ Cleanup completed${NC}"
    else
        echo "  docker stop $CONTAINER_NAME"
        echo "  docker rm $CONTAINER_NAME"
        echo "  docker rmi $IMAGE_NAME:latest"
    fi
}

# Main menu
main() {
    echo "========================================="
    echo "Docker Container Management Demo"
    echo "========================================="
    echo ""
    
    check_docker
    echo ""
    
    build_image
    echo ""
    
    list_images
    echo ""
    
    run_container
    echo ""
    
    list_containers
    echo ""
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        sleep 2
        view_logs
        echo ""
    fi
    
    echo "Press Enter to stop and cleanup, or Ctrl+C to keep running"
    read -r
    
    stop_container
    echo ""
    
    cleanup
}

# Run the script
main
