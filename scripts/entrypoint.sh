#!/bin/bash
set -e

# S3_BUCKET=s3://image-captioning-models
# MODEL_NAME=caption_model_v1.pth
# DEST_PATH=inference/models/model.pth

# mkdir -p inference/artifacts

# echo "Downloading model from S3..."
# aws s3 cp $S3_BUCKET/$MODEL_NAME $DEST_PATH

echo "Model ready for inference."

echo "Running startup tasks..."

echo "Starting FastAPI..."
exec uvicorn inference.app:app --host 0.0.0.0 --port 8000
