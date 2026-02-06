#!/bin/bash
# Deploy script for Model Lab production API
# Usage: ./scripts/deploy.sh [build|run|push|health]
#
# Environment variables:
#   REGISTRY: Docker registry URL (default: none, local only)
#   TAG: Image tag (default: latest)
#   PORT: Port to expose (default: 8000)
#   MODEL_CACHE_MAX_MB: Max memory for model cache (default: 4096)

set -euo pipefail

# Configuration
IMAGE_NAME="model-lab"
TAG="${TAG:-latest}"
PORT="${PORT:-8000}"
REGISTRY="${REGISTRY:-}"
FULL_IMAGE="${REGISTRY:+$REGISTRY/}${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cmd_build() {
    log_info "Building Docker image: $FULL_IMAGE"
    cd "$PROJECT_ROOT"
    docker build -t "$FULL_IMAGE" .
    log_info "Build complete: $FULL_IMAGE"
}

cmd_run() {
    log_info "Running container on port $PORT"
    docker run -d \
        --name model-lab-api \
        -p "${PORT}:8000" \
        -e MODEL_CACHE_MAX_MB="${MODEL_CACHE_MAX_MB:-4096}" \
        -e LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        --restart unless-stopped \
        "$FULL_IMAGE"
    
    log_info "Container started. Waiting for health check..."
    sleep 5
    cmd_health
}

cmd_push() {
    if [ -z "$REGISTRY" ]; then
        log_error "REGISTRY not set. Cannot push."
        exit 1
    fi
    log_info "Pushing image to registry: $FULL_IMAGE"
    docker push "$FULL_IMAGE"
    log_info "Push complete"
}

cmd_health() {
    log_info "Checking health endpoint..."
    local url="http://localhost:${PORT}/health"
    
    if curl -sf "$url" > /dev/null 2>&1; then
        log_info "Health check passed ✓"
        curl -s "$url" | python3 -m json.tool
    else
        log_error "Health check failed"
        exit 1
    fi
}

cmd_stop() {
    log_info "Stopping container..."
    docker stop model-lab-api 2>/dev/null || true
    docker rm model-lab-api 2>/dev/null || true
    log_info "Container stopped"
}

cmd_logs() {
    docker logs -f model-lab-api
}

cmd_rollback() {
    log_warn "Rolling back to previous image..."
    local prev_tag="${TAG}-prev"
    local prev_image="${REGISTRY:+$REGISTRY/}${IMAGE_NAME}:${prev_tag}"
    
    if docker image inspect "$prev_image" > /dev/null 2>&1; then
        cmd_stop
        TAG="$prev_tag" cmd_run
        log_info "Rollback complete"
    else
        log_error "No previous image found: $prev_image"
        exit 1
    fi
}

cmd_help() {
    cat << EOF
Model Lab Deployment Script

Usage: $0 <command>

Commands:
  build     Build Docker image
  run       Run container locally
  push      Push image to registry (requires REGISTRY env var)
  health    Check health endpoint
  stop      Stop running container
  logs      Follow container logs
  rollback  Rollback to previous image version

Environment Variables:
  REGISTRY            Docker registry URL
  TAG                 Image tag (default: latest)
  PORT                Port to expose (default: 8000)
  MODEL_CACHE_MAX_MB  Max memory for model cache (default: 4096)
  LOG_LEVEL           Logging level (default: INFO)

Examples:
  $0 build                    # Build image locally
  $0 run                      # Run on default port
  PORT=9000 $0 run           # Run on custom port
  REGISTRY=ghcr.io/user $0 push  # Push to registry
EOF
}

# Main
case "${1:-help}" in
    build)   cmd_build ;;
    run)     cmd_run ;;
    push)    cmd_push ;;
    health)  cmd_health ;;
    stop)    cmd_stop ;;
    logs)    cmd_logs ;;
    rollback) cmd_rollback ;;
    help|--help|-h) cmd_help ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
