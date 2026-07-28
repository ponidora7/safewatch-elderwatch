---
title: SafeWatch Backend AI
emoji: 👴🏻🚨
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SafeWatch Backend API (Hugging Face Space)

This is the FastAPI backend for the SafeWatch ElderWatch project. 
It performs real-time fall detection using **YOLOv8-Pose** and a custom **ONNX Knowledge Distillation** model.

## Why Hugging Face Spaces?
This backend is optimized to run on Hugging Face Spaces (Docker SDK), which provides generous free-tier resources (16GB RAM, 2 vCPUs) that are essential for loading PyTorch and ONNX Runtime models simultaneously without hitting Out-Of-Memory (OOM) errors.

## Endpoints
- `GET /health` : Check if the models are loaded.
- `POST /pose-extract` : Extracts 17 keypoints + bounding box from base64 image (used for client-side ONNX logic).
- `POST /inference` : Full backend fallback inference.
- `POST /log-incident` : Client-side reporting of confirmed falls to Supabase.

## Setup Instructions for Deployment
1. Create a new **Docker Space** on Hugging Face.
2. Copy the contents of this `backend/` directory into the root of your Hugging Face Space repository.
3. Make sure to copy the model files from the root `models/` directory into a `models/` directory inside your Hugging Face space:
   - `models/safewatch_model_cpu.onnx`
   - `models/feature_scaler.pkl`
   - `models/yolov8n-pose.pt` (or let it auto-download)
4. Add your secrets in Hugging Face Space Settings (Variables and secrets):
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `RESEND_API_KEY` (optional for emails)
   - `ALERT_EMAIL`
