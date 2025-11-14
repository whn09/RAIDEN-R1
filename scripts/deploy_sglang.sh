#!/bin/bash
# Deploy SGLang server for RAIDEN data generation
# Supports various open-source models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================="
echo "RAIDEN-R1 SGLang Deployment"
echo -e "==================================${NC}"

# Parse arguments
MODEL_PATH=""
MODEL_NAME=""
PORT=30000
TP_SIZE=8  # Tensor parallel size for p5en.48xlarge (8 H100 GPUs)
HOST="0.0.0.0"
MEMORY_FRACTION=0.9

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --memory-fraction)
            MEMORY_FRACTION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: --model-path is required${NC}"
    echo ""
    echo "Usage: $0 --model-path /path/to/model [options]"
    echo ""
    echo "Options:"
    echo "  --model-path PATH       Path to model (required)"
    echo "  --model-name NAME       Model name for API (optional)"
    echo "  --port PORT             Server port (default: 30000)"
    echo "  --tp-size SIZE          Tensor parallel size (default: 8 for p5en.48xlarge)"
    echo "  --memory-fraction NUM   GPU memory fraction (default: 0.9)"
    echo ""
    echo "Examples:"
    echo "  # Deploy MiniMax M2"
    echo "  $0 --model-path /models/MiniMax-Text-01 --model-name minimax-m2"
    echo ""
    echo "  # Deploy GLM-4"
    echo "  $0 --model-path /models/glm-4-9b --model-name glm-4 --tp-size 4"
    echo ""
    echo "  # Deploy Qwen2.5-14B"
    echo "  $0 --model-path /models/Qwen2.5-14B-Instruct --model-name qwen2.5-14b --tp-size 4"
    exit 1
fi

# Set model name if not provided
if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH")
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Model Path: $MODEL_PATH"
echo "  Model Name: $MODEL_NAME"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $TP_SIZE"
echo "  Host: $HOST"
echo "  Memory Fraction: $MEMORY_FRACTION"
echo ""

# Check if SGLang is installed
if ! python -c "import sglang" 2>/dev/null; then
    echo -e "${YELLOW}SGLang not found. Installing...${NC}"
    pip install "sglang[all]" --quiet
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path does not exist: $MODEL_PATH${NC}"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. GPU required for SGLang.${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}Detected $GPU_COUNT GPUs${NC}"

if [ "$TP_SIZE" -gt "$GPU_COUNT" ]; then
    echo -e "${YELLOW}Warning: TP_SIZE ($TP_SIZE) > GPU_COUNT ($GPU_COUNT)${NC}"
    echo -e "${YELLOW}Setting TP_SIZE to $GPU_COUNT${NC}"
    TP_SIZE=$GPU_COUNT
fi

# Kill existing SGLang server on this port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}Killing existing server on port $PORT...${NC}"
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start SGLang server
echo ""
echo -e "${GREEN}Starting SGLang server...${NC}"
echo -e "${YELLOW}This may take a few minutes to load the model...${NC}"
echo ""

# Create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sglang_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Start server in background
nohup python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size "$TP_SIZE" \
    --mem-fraction-static "$MEMORY_FRACTION" \
    --enable-torch-compile \
    --context-length 32768 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Log file: $LOG_FILE"

# Wait for server to start
echo ""
echo -e "${YELLOW}Waiting for server to start...${NC}"
MAX_WAIT=300  # 5 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready!${NC}"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo -n "."
done
echo ""

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}✗ Server failed to start within ${MAX_WAIT} seconds${NC}"
    echo -e "${YELLOW}Check log file: $LOG_FILE${NC}"
    exit 1
fi

# Test the server
echo ""
echo -e "${GREEN}Testing server...${NC}"
TEST_RESPONSE=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_NAME"'",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }')

if echo "$TEST_RESPONSE" | grep -q "choices"; then
    echo -e "${GREEN}✓ Server test passed!${NC}"
else
    echo -e "${RED}✗ Server test failed${NC}"
    echo "Response: $TEST_RESPONSE"
    exit 1
fi

# Save server info
SERVER_INFO_FILE="./sglang_server_info.json"
cat > "$SERVER_INFO_FILE" <<EOF
{
  "model_name": "$MODEL_NAME",
  "model_path": "$MODEL_PATH",
  "base_url": "http://localhost:$PORT",
  "pid": $SERVER_PID,
  "tp_size": $TP_SIZE,
  "started_at": "$(date -Iseconds)",
  "log_file": "$LOG_FILE"
}
EOF

echo ""
echo -e "${GREEN}=================================="
echo "SGLang Server Started Successfully!"
echo -e "==================================${NC}"
echo ""
echo "Server Information:"
echo "  Base URL: http://localhost:$PORT"
echo "  Model: $MODEL_NAME"
echo "  PID: $SERVER_PID"
echo "  Log: $LOG_FILE"
echo ""
echo "Server info saved to: $SERVER_INFO_FILE"
echo ""
echo "To generate data with this server:"
echo -e "  ${YELLOW}python scripts/generate_data_with_sglang.py \\${NC}"
echo -e "    ${YELLOW}--base-url http://localhost:$PORT \\${NC}"
echo -e "    ${YELLOW}--model-name $MODEL_NAME${NC}"
echo ""
echo "To stop the server:"
echo "  kill $SERVER_PID"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
