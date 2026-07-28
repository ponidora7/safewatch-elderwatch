# SafeWatch ElderWatch

AI-powered monitoring system for elderly care, detecting falls and hazards via computer vision.

## Project Structure

- `backend/`: FastAPI application providing AI inference relay, Supabase integration, and notifications.
- `frontend/`: Next.js 15 dashboard for real-time monitoring and alerts.
- `models/`: Pre-trained models and classifier scripts (legacy/reference).
- `scripts/`: Utility scripts for training and evaluation.

## Getting Started

### Backend Setup

1. Navigate to `backend/`
2. Create a `.env` file based on `.env.example`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `uvicorn main:app --reload`

### Frontend Setup

1. Navigate to `frontend/`
2. Create a `.env.local` file based on `.env.example`
3. Install dependencies: `pnpm install`
4. Run the development server: `pnpm run dev`

## Deployment

- **Backend**: Deployed to Railway via GitHub Actions.
- **Frontend**: Deployed to Cloudflare Pages (Static Export) via GitHub Actions.

## Tech Stack

- **Backend**: FastAPI, Supabase, Resend, Hugging Face.
- **Frontend**: Next.js 15, Tailwind CSS, shadcn/ui, Supabase JS.
