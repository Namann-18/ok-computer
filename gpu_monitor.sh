#!/bin/bash

# ================================================================================
# GPU Monitoring Script for Training
# ================================================================================
# Monitors GPU usage and detects crashes during YOLO training
# ================================================================================

# Configuration
CHECK_INTERVAL=10
HIGH_MEMORY_THRESHOLD=14000  # MB
CRASH_DETECTION_WINDOW=60    # seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to get GPU info
get_gpu_info() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
}

# Function to check for GPU processes
check_training_process() {
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits | grep -i python
}

# Function to log GPU status
log_gpu_status() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local gpu_info=$(get_gpu_info)
    local processes=$(check_training_process)
    
    echo "[$timestamp] GPU Status:"
    echo "$gpu_info" | while IFS=',' read -r idx name mem_used mem_total util temp; do
        echo "  GPU $idx: $name"
        echo "    Memory: ${mem_used}MB / ${mem_total}MB ($(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l)%)"
        echo "    Utilization: ${util}%"
        echo "    Temperature: ${temp}°C"
        
        # Check for high memory usage
        if [ "$mem_used" -gt "$HIGH_MEMORY_THRESHOLD" ]; then
            echo -e "    ${RED}WARNING: High memory usage!${NC}"
        fi
    done
    
    if [ -n "$processes" ]; then
        echo "  Training processes:"
        echo "$processes" | while IFS=',' read -r pid name gpu_uuid mem; do
            echo "    PID $pid: $name (GPU: $gpu_uuid, Memory: ${mem}MB)"
        done
    else
        echo -e "  ${YELLOW}No training processes detected${NC}"
    fi
    echo "----------------------------------------"
}

# Function to detect crashes
detect_crash() {
    local last_check=$(date +%s)
    local no_process_count=0
    
    while true; do
        local current_time=$(date +%s)
        local processes=$(check_training_process)
        
        if [ -z "$processes" ]; then
            no_process_count=$((no_process_count + CHECK_INTERVAL))
            if [ $no_process_count -ge $CRASH_DETECTION_WINDOW ]; then
                echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] CRASH DETECTED: No training processes for ${CRASH_DETECTION_WINDOW}s${NC}"
                return 0
            fi
        else
            no_process_count=0
        fi
        
        sleep $CHECK_INTERVAL
    done
}

# Main monitoring loop
main() {
    echo -e "${GREEN}Starting GPU monitoring...${NC}"
    echo "Check interval: ${CHECK_INTERVAL}s"
    echo "High memory threshold: ${HIGH_MEMORY_THRESHOLD}MB"
    echo "Crash detection window: ${CRASH_DETECTION_WINDOW}s"
    echo "========================================"
    
    # Initial GPU status
    log_gpu_status
    
    # Start monitoring loop
    while true; do
        sleep $CHECK_INTERVAL
        log_gpu_status
        
        # Check for crashes in background
        detect_crash &
        crash_pid=$!
        
        # Wait for next check or crash detection
        sleep $CHECK_INTERVAL
        
        # Check if crash was detected
        if ! kill -0 $crash_pid 2>/dev/null; then
            echo -e "${RED}Training crash detected!${NC}"
            break
        fi
    done
}

# Handle script interruption
trap 'echo -e "\n${YELLOW}GPU monitoring stopped.${NC}"; exit 0' INT TERM

# Run main function
main "$@"

