# Outputs Directory

This directory contains training outputs, checkpoints, and predictions.

## Structure

```
outputs/
├── checkpoints/        # Model checkpoints (.pth files)
├── logs/              # Training logs and history
└── predictions/       # Inference results and visualizations
```

## Notes

- Checkpoints are saved every N epochs (configurable)
- Best model is saved as `best_model.pth`
- Training history is logged to `logs/training_history.json`
- Predictions include both metrics and visualizations
