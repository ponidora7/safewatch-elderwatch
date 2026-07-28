# SafeWatch ElderWatch — Frontend

Sistem pemantauan keselamatan lansia berbasis AI menggunakan deteksi jatuh real-time.

## Teknologi

- **Vite 6** + **React 19** + **TypeScript 5.8**
- **ONNX Runtime Web** — inferensi model fall detection langsung di browser (WebAssembly)
- **Supabase Realtime** — notifikasi insiden via WebSocket
- **Tailwind CSS 4** — styling

## Cara Menjalankan

**Prerequisites:** Node.js 18+ / pnpm

1. Install dependencies:
   ```bash
   pnpm install
   ```

2. Salin file environment dan isi nilainya:
   ```bash
   cp .env.example .env.local
   ```
   
   Isi nilai berikut di `.env.local`:
   ```env
   VITE_API_URL=http://localhost:8000
   VITE_SUPABASE_URL=https://your-project.supabase.co
   VITE_SUPABASE_KEY=your-supabase-anon-key
   ```

3. Pastikan model ONNX sudah ada di `public/models/`:
   - `public/models/fall_model.onnx`
   - `public/models/scaler.json`
   
   Jika belum ada, jalankan dari root project:
   ```bash
   python scripts/train_distilled_model.py
   ```

4. Jalankan development server:
   ```bash
   pnpm dev
   ```
   
   Buka: `http://localhost:3000`

## Arsitektur Inference

```
Webcam frame (1500ms interval)
    │
    ▼
POST /pose-extract  (Backend FastAPI)
    │  YOLOv8-Pose → 16 koordinat + raw_features
    ▼
onnxInference.ts  (Browser ONNX.js / WebAssembly)
    │  StandardScaler normalization + 35 fitur → fall_model.onnx
    ▼
Weighted temporal smoothing (7 frame, bobot 0.5–2.0)
    │
    ▼
Fall Detected? → POST /log-incident → Supabase + Email alert
```
