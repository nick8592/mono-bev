#!/bin/bash
# Script to monitor training progress

echo "=== Training Status ==="
echo ""

# Check if training is running
if pgrep -f "python train.py" > /dev/null; then
    echo "✅ Training is running (PID: $(pgrep -f 'python train.py'))"
else
    echo "❌ Training is not running"
fi

echo ""
echo "=== Latest Training Output ==="
tail -n 50 training.log

echo ""
echo "=== Checkpoints ==="
ls -lh checkpoints/ 2>/dev/null || echo "No checkpoints yet"

echo ""
echo "=== GPU Memory Usage ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "Could not query GPU"
