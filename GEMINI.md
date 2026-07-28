# SafeWatch ElderWatch: Antigravity IDE Implementation Plan

This plan is optimized for **Antigravity IDE**, leveraging its agent-first architecture to accelerate the transition of SafeWatch to the "Free-First Stack." By using Antigravity, you can delegate complex refactoring and setup tasks to your internal agents.

---

## 🚀 Antigravity Workflow Strategy

In Antigravity, you should treat the implementation as a series of **Agent Missions**. Instead of writing every line of code, you will guide your Antigravity agent using the prompts provided in each phase below.

### Phase 1: Environment & Account Setup (Agent-Assisted)
**Goal:** Establish the cloud infrastructure and repository structure.

1.  **Cloud Setup:** Manually create accounts for Supabase, Railway, Hugging Face, and Cloudflare as these require OAuth/Browser interaction.
2.  **Workspace Initialization:** Use the Antigravity terminal to clone the repo and organize the folders.
3.  **Antigravity Prompt:**
    > "Initialize the workspace for SafeWatch. Create a clear directory structure separating `/frontend` (Next.js) and `/backend` (FastAPI). Move existing Streamlit code to `/legacy/dashboard` and existing ML scripts to `/legacy/ml`. Set up a `.env.example` file with placeholders for Supabase, Railway, and Resend keys."

---

### Phase 2: AI Model Optimization (Agent Mission)
**Goal:** Convert the enhanced model to a CPU-optimized format for Hugging Face Spaces.

1.  **Optimization Script:** Let the agent handle the ONNX conversion and quantization logic.
2.  **Antigravity Prompt:**
    > "Read `models/safewatch_fall_model_enhanced.keras`. Create a Python script using `tf2onnx` to convert it to ONNX format. Then, use `onnxruntime.quantization` to apply dynamic INT8 quantization. Benchmark the inference speed on a sample image and ensure it's under 500ms. Save the optimized model as `models/safewatch_model_cpu.onnx`."

---

### Phase 3: Backend API Development (FastAPI)
**Goal:** Build the central hub on Railway.

1.  **Core Logic:** The agent can scaffold the FastAPI structure and integrate Supabase.
2.  **Antigravity Prompt:**
    > "Build a FastAPI backend in the `/backend` directory. Implement a `POST /inference` endpoint that: 1. Receives a base64 image. 2. Calls the Hugging Face Inference API. 3. Logs the incident in Supabase if a fall is detected. Integrate `supabase-py` for database operations and use `Resend` for email alerts. Ensure there is a `/health` endpoint for keep-alive pings."

---

### Phase 4: Frontend Migration (Next.js 15)
**Goal:** Replace the Streamlit dashboard with a high-performance Next.js app on Cloudflare Pages.

1.  **UI Scaffolding:** Use Antigravity's ability to generate React components.
2.  **Antigravity Prompt:**
    > "Initialize a Next.js 15 project in `/frontend` using Tailwind CSS and shadcn/ui. Create a 'Live Monitor' page that: 1. Accesses the user's webcam. 2. Captures a frame every 2 seconds. 3. Sends it to the backend `/inference` endpoint. 4. Subscribes to Supabase Realtime for instant 'Fall Detected' alerts. Ensure the app is configured for Static Export."

---

### Phase 5: Integration & Keep-Alive (Final Polish)
**Goal:** Connect all pieces and ensure zero-cost uptime.

1.  **Realtime Testing:** Use the agent to verify the WebSocket connection between Supabase and the Frontend.
2.  **Antigravity Prompt:**
    > "Write an integration test script that simulates a fall detection event in the database and verifies that the Frontend receives the Realtime broadcast within 2 seconds. Also, create a `keep_alive.py` script that pings the Railway and Hugging Face endpoints, which I can later host on cron-job.org."

---

## 🛠️ Key Antigravity Shortcuts for this Project
*   **Cmd/Ctrl + L:** Use this to ask the agent to explain any legacy Python script in the repository before refactoring.
*   **Agent Browser:** If you get stuck with Supabase RLS policies, ask the agent: *"Open the Supabase documentation in the browser and find the correct RLS policy for a public-read/private-write incidents table."*
*   **Terminal Sync:** Antigravity agents can run `pip install` or `npm install` automatically. Always verify the `requirements.txt` and `package.json` they generate.

---

## ⚠️ Critical Reminder for Free Tier
*   **Cold Starts:** Remind your agent to implement a "Loading State" in the frontend. When the backend is "sleeping" (Railway/HF free tier), the UI should show a "Waking up AI services..." message to the user.
*   **Database Limits:** Ensure the agent implements a cleanup task or a limit on how many screenshots are saved to Cloudflare R2 to stay within the 10GB free limit.







# SafeWatch Project - AI Assistant Guidelines

Welcome to the SafeWatch project repository. When operating within this workspace, please adhere to the following architecture, conventions, and workflows.

## 1. Project Context & Documentation
- **Primary Goal:** A dual-purpose computer vision system for Fall Detection (Keras/TensorFlow) and Fire/Smoke Detection (YOLOv8).
- **Start Here:** Always read `DOCUMENTATION_INDEX.md` and `IMPLEMENTATION_SUMMARY.md` before starting a new complex task to understand current architectural decisions and recent changes.
- **Project Structure:** Reference `file_structure.md` or `workspace_structure.txt` for the layout.

## 2. Technical Stack
- **Languages:** Python (Primary for ML pipelines/APIs), PowerShell/Batch (DevOps/Setup).
- **Core Libraries:**
  - TensorFlow / Keras (Fall Detection Model)
  - Ultralytics YOLO (Fire & Smoke Detection Model)
  - OpenCV, MediaPipe (Video processing and Pose extraction)
  - Scikit-learn, Pandas, NumPy (Data handling)

## 3. Engineering & AI Rules
- **No Direct Edits to Raw Data:** Do not attempt to modify anything in `data/raw/` or `data/real_footage/`.
- **Model Files:** Pre-trained or trained models are saved in `models/`. Core models are tracked in git (`!models/...` in `.gitignore`), do not arbitrarily delete them.
- **Idempotent Scripts:** Scripts in `scripts/` (e.g., `assess_all.py`, `train_fall.py`) should be designed to be run multiple times safely.
- **Python Conventions:**
  - Follow PEP8 style guidelines.
  - Use type hints wherever possible for new functions.
  - Rely on `config/paths.py` for all file path resolving to maintain cross-platform compatibility. Do not hardcode paths.
- **Notebooks:** `.ipynb` files in `notebooks/` are for exploration (EDA) and experimental training. Production pipeline code should live in `src/` and `scripts/`.

## 4. Workflows
- **Data Pipeline:** Handled in `src/` (`data_cleaner.py`, `feature_temporal.py`, etc.).
- **Testing:** Before suggesting changes to the model inference logic, refer to `TEST_MODEL_GUIDE.md`.

## 5. Environment & Git
- Do not commit `.env` files or large untracked model files unless specifically instructed.
- When generating reports or large output logs, save them to the `reports/` folder, which is appropriately managed by `.gitignore`.
