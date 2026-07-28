/**
 * SafeWatch — On-Device ONNX Fall Classifier (Browser-side)
 * ==========================================================
 * Loads the fall detection ONNX model and feature scaler directly in the browser.
 * This eliminates cold-start delays from the backend for classification.
 *
 * Architecture:
 *   Backend (/pose-extract) → raw_features (16 floats)
 *   → [THIS FILE] → normalize → ONNX.js → fall probability
 *   → Frontend temporal smoothing → alert
 */

import * as ort from 'onnxruntime-web'

// ---- Types ----------------------------------------------------------------

export interface ScalerParams {
  mean_: number[]
  scale_: number[]
  n_features_in_: number
}

export interface GeometricFeatures {
  torso_ratio: number
  left_knee_angle: number
  right_knee_angle: number
  hip_drop: number
  body_lean: number
  shoulder_width: number
  hip_width: number
  leg_spread: number
  left_arm_angle: number
  right_arm_angle: number
  hip_height: number
  knee_height: number
  ankle_height: number
  vertical_extent: number
  horizontal_extent: number
  aspect_ratio: number
  center_y: number
  center_x: number
  left_hip_knee_angle: number
}

export interface FallPrediction {
  detected: boolean
  confidence: number
  inference_ms: number
  mode: 'onnx_browser' | 'server_fallback'
}

// ---- State (module-level singletons) -------------------------------------

let session: ort.InferenceSession | null = null
let scaler: ScalerParams | null = null
let isLoading = false
let loadError: string | null = null

// ---- Loaders --------------------------------------------------------------

/**
 * Initialize the ONNX model and scaler. Safe to call multiple times.
 * Returns true if both loaded successfully.
 */
export async function initOnnxModel(): Promise<boolean> {
  if (session && scaler) return true
  if (isLoading) {
    // Wait for ongoing load
    await new Promise<void>((resolve) => {
      const check = setInterval(() => {
        if (!isLoading) { clearInterval(check); resolve() }
      }, 100)
    })
    return session !== null && scaler !== null
  }

  isLoading = true
  loadError = null

  try {
    // Configure ONNX Runtime WebAssembly via CDN to guarantee it loads
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'
    ort.env.wasm.numThreads = 1

    // Load model
    const modelUrl = '/models/fall_model.onnx'
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
    console.log('[ONNX] Fall model loaded successfully')
    console.log('[ONNX] Input:', session.inputNames)
    console.log('[ONNX] Output:', session.outputNames)

    // Load scaler params
    const scalerRes = await fetch('/models/scaler.json')
    if (!scalerRes.ok) throw new Error(`Scaler fetch failed: ${scalerRes.status}`)
    scaler = await scalerRes.json() as ScalerParams
    console.log(`[ONNX] Scaler loaded — ${scaler.n_features_in_} features`)

    isLoading = false
    return true
  } catch (err) {
    const errMsg = String(err)
    // Suppress noisy ONNX model format errors — geometric fallback will handle detection
    if (errMsg.includes('getValue') || errMsg.includes('InferenceSession')) {
      console.warn('[ONNX] Model format incompatible with browser runtime — using geometric detection instead.')
    } else {
      console.error('[ONNX] Failed to load model/scaler:', err)
    }
    loadError = errMsg
    isLoading = false
    session = null
    scaler = null
    return false
  }
}

export function getOnnxStatus(): { ready: boolean; error: string | null; loading: boolean } {
  return { ready: session !== null && scaler !== null, error: loadError, loading: isLoading }
}

// ---- Feature Engineering --------------------------------------------------

/**
 * Extract geometric features from 16 raw landmark values (8 joints × [x,y]).
 * Mirrors the Python AdvancedFeatureEngineer.extract_geometric_features().
 * Joint order: [shoulder_L, shoulder_R, hip_L, hip_R, knee_L, knee_R, ankle_L, ankle_R]
 */
function extractGeometricFeatures(landmarks: number[]): number[] {
  const [
    sh_lx, sh_ly, sh_rx, sh_ry,
    hp_lx, hp_ly, hp_rx, hp_ry,
    kn_lx, kn_ly, kn_rx, kn_ry,
    an_lx, an_ly, an_rx, an_ry
  ] = landmarks

  const dist = (ax: number, ay: number, bx: number, by: number) =>
    Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

  const angle = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => {
    const v1x = ax - bx, v1y = ay - by
    const v2x = cx - bx, v2y = cy - by
    const mag1 = Math.sqrt(v1x ** 2 + v1y ** 2)
    const mag2 = Math.sqrt(v2x ** 2 + v2y ** 2)
    if (mag1 === 0 || mag2 === 0) return 0.0
    const dot = v1x * v2x + v1y * v2y
    let cos_angle = dot / (mag1 * mag2)
    cos_angle = Math.max(-1, Math.min(1, cos_angle))
    return Math.acos(cos_angle) * (180 / Math.PI)
  }

  const torso_top_x = (sh_lx + sh_rx) / 2
  const torso_top_y = (sh_ly + sh_ry) / 2
  const torso_bot_x = (hp_lx + hp_rx) / 2
  const torso_bot_y = (hp_ly + hp_ry) / 2
  
  const torso_vec_x = torso_bot_x - torso_top_x
  const torso_vec_y = torso_bot_y - torso_top_y
  const torso_mag = Math.sqrt(torso_vec_x ** 2 + torso_vec_y ** 2)
  
  let torso_angle = 0.0
  let torso_tilt = 0.0
  if (torso_mag > 0) {
      // angle with vertical [0, 1] (y goes down)
      let cos_angle = torso_vec_y / torso_mag
      cos_angle = Math.max(-1, Math.min(1, cos_angle))
      torso_angle = Math.acos(cos_angle) * (180 / Math.PI)
      torso_tilt = Math.abs(torso_top_x - torso_bot_x) / torso_mag
  }
  
  const spine_curvature = Math.abs(torso_top_x - torso_bot_x)
  const torso_length = torso_mag + 1e-6
  const left_leg_len = dist(hp_lx, hp_ly, an_lx, an_ly)
  const right_leg_len = dist(hp_rx, hp_ry, an_rx, an_ry)
  const avg_leg_len = (left_leg_len + right_leg_len) / 2
  const hip_width = dist(hp_lx, hp_ly, hp_rx, hp_ry)
  const shoulder_width = dist(sh_lx, sh_ly, sh_rx, sh_ry)
  const body_height = torso_length + avg_leg_len
  const body_width = Math.max(hip_width, shoulder_width, 1.0)
  const ankle_spread = Math.abs(an_lx - an_rx)
  
  const left_knee_angle = angle(hp_lx, hp_ly, kn_lx, kn_ly, an_lx, an_ly)
  const right_knee_angle = angle(hp_rx, hp_ry, kn_rx, kn_ry, an_rx, an_ry)
  const is_horizontal = torso_angle > 60 ? 1.0 : 0.0
  
  const leg_vec_lx = an_lx - kn_lx, leg_vec_ly = an_ly - kn_ly
  const leg_vec_rx = an_rx - kn_rx, leg_vec_ry = an_ry - kn_ry
  const mag_leg_l = Math.sqrt(leg_vec_lx ** 2 + leg_vec_ly ** 2)
  const mag_leg_r = Math.sqrt(leg_vec_rx ** 2 + leg_vec_ry ** 2)
  
  let leg_spread_angle = 0.0
  if (mag_leg_l > 0 && mag_leg_r > 0) {
      const dot = leg_vec_lx * leg_vec_rx + leg_vec_ly * leg_vec_ry
      let cos_leg = dot / (mag_leg_l * mag_leg_r)
      cos_leg = Math.max(-1, Math.min(1, cos_leg))
      leg_spread_angle = Math.acos(cos_leg) * (180 / Math.PI)
  }
  
  const com_y = (torso_top_y + torso_bot_y) / 2
  const com_x = (an_lx + an_rx) / 2
  
  // 19 geometric features
  const features = [
    angle(sh_lx, sh_ly, hp_lx, hp_ly, kn_lx, kn_ly), // left_hip_angle
    angle(sh_rx, sh_ry, hp_rx, hp_ry, kn_rx, kn_ry), // right_hip_angle
    left_knee_angle,
    right_knee_angle,
    torso_angle,
    torso_tilt,
    spine_curvature,
    torso_length / (avg_leg_len + 1e-6), // torso_to_leg_ratio
    hip_width / torso_length,            // hip_width_ratio
    shoulder_width / torso_length,       // shoulder_width_ratio
    body_height / body_width,            // body_aspect_ratio
    ankle_spread / torso_length,         // ankle_spread_ratio
    is_horizontal,
    leg_spread_angle,
    Math.abs(sh_ly - sh_ry) / torso_length, // shoulder_symmetry
    (left_knee_angle + right_knee_angle) / 2, // avg_knee_angle
    (com_x - torso_bot_x) / torso_length, // com_x_norm
    (com_y - torso_bot_y) / torso_length  // com_y_norm
  ]
  
  // 16 biologically normalized base coordinates
  const scale = torso_length
  const hip_cx = torso_bot_x, hip_cy = torso_bot_y
  
  const normCoords = [
    (sh_lx - hip_cx) / scale, (sh_ly - hip_cy) / scale, // norm_X11, norm_Y11
    (sh_rx - hip_cx) / scale, (sh_ry - hip_cy) / scale, // norm_X12, norm_Y12
    (hp_lx - hip_cx) / scale, (hp_ly - hip_cy) / scale, // norm_X23, norm_Y23
    (hp_rx - hip_cx) / scale, (hp_ry - hip_cy) / scale, // norm_X24, norm_Y24
    (kn_lx - hip_cx) / scale, (kn_ly - hip_cy) / scale, // norm_X25, norm_Y25
    (kn_rx - hip_cx) / scale, (kn_ry - hip_cy) / scale, // norm_X26, norm_Y26
    (an_lx - hip_cx) / scale, (an_ly - hip_cy) / scale, // norm_X27, norm_Y27
    (an_rx - hip_cx) / scale, (an_ry - hip_cy) / scale  // norm_X28, norm_Y28
  ]
  
  return [...features, ...normCoords]
}

/**
 * Normalize features using StandardScaler params loaded from JSON.
 */
function scalerTransform(features: number[]): number[] {
  if (!scaler) return features
  return features.map((v, i) => (v - scaler!.mean_[i]) / (scaler!.scale_[i] + 1e-9))
}

// ---- Main Inference -------------------------------------------------------

/**
 * Run fall classification on raw_features received from /pose-extract.
 * @param rawFeatures - 16 floats [shoulder_L_x, shoulder_L_y, ..., ankle_R_y]
 * @returns FallPrediction with confidence and detected flag
 */
export async function runFallClassifier(rawFeatures: number[]): Promise<FallPrediction | null> {
  if (!session || !scaler) {
    const loaded = await initOnnxModel()
    if (!loaded) return null
  }

  const t0 = performance.now()

  try {
    // 1. Extract biological normalized features (34 total)
    const allFeatures = extractGeometricFeatures(rawFeatures)

    // 3. Scale features
    const scaled = scalerTransform(allFeatures)

    // 4. Create ONNX tensor
    const inputTensor = new ort.Tensor('float32', Float32Array.from(scaled), [1, 34])

    // 5. Run inference
    const inputName = session!.inputNames[0]
    const outputs = await session!.run({ [inputName]: inputTensor })

    // 6. Get output probability
    const outputName = session!.outputNames[0]
    const outputData = outputs[outputName].data as Float32Array
    const confidence = outputData[0]

    const inference_ms = Math.round(performance.now() - t0)

    return {
      detected: confidence >= 0.65,  // Default threshold (matches backend)
      confidence,
      inference_ms,
      mode: 'onnx_browser'
    }
  } catch (err) {
    console.error('[ONNX] Inference error:', err)
    return null
  }
}
