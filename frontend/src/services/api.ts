/**
 * SafeWatch API Service
 * Supports hybrid inference: primary pose-extract + client-side ONNX,
 * with fallback to full server-side inference (/inference endpoint).
 */

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8000'

// ---- Types ----------------------------------------------------------------

export interface PoseExtractResponse {
  status: string
  keypoints: number[][] | null // Array of [x, y] coordinates
  box: number[] | null         // [ymin, xmin, ymax, xmax]
  raw_features: number[] | null
  person_detected: boolean
  hazards?: any[]              // Array of fire/smoke hazards
  reason?: string
}

export interface InferenceResponse {
  status: string
  detected: boolean
  confidence: number
  ai_result?: {
    detected: boolean
    confidence: number
    type: string
    inference_ms: number
    keypoints: number[][] | null
    box: number[] | null
  }
}

// ---- Helper ---------------------------------------------------------------

async function apiFetch<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(`API ${path} returned ${response.status}`)
  }
  return response.json() as Promise<T>
}

// ---- Endpoints ------------------------------------------------------------

/**
 * HYBRID MODE (Recommended):
 * Extract pose keypoints from server. Classification done client-side via ONNX.js.
 */
export async function extractPose(imageBase64: string): Promise<PoseExtractResponse> {
  return apiFetch<PoseExtractResponse>('/pose-extract', { image: imageBase64 })
}

/**
 * FALLBACK MODE:
 * Full server-side inference (YOLOv8 + ONNX fall classifier on server).
 * Used when browser ONNX.js is unavailable or fails.
 */
export async function sendFrame(imageBase64: string): Promise<InferenceResponse> {
  return apiFetch<InferenceResponse>('/inference', { image: imageBase64 })
}

/**
 * Log a confirmed fall event to Supabase via backend (with email alert).
 * Called after client-side temporal smoothing confirms a fall.
 */
export async function logIncident(data: {
  type: string
  confidence: number
  inference_ms: number
  source?: string
}): Promise<void> {
  try {
    await apiFetch('/log-incident', { ...data, source: data.source ?? 'client_onnx' })
  } catch (err) {
    console.error('[API] Failed to log incident:', err)
  }
}

/**
 * Health check — verify backend is awake and models are loaded.
 */
export async function checkHealth(): Promise<{
  status: string
  models?: { yolo_pose: boolean; fall_classifier: boolean; scaler: boolean }
}> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`)
    return res.json()
  } catch (err) {
    console.error('[API] Health check failed:', err)
    throw err
  }
}
