/**
 * WebcamFeed — Hybrid AI Monitor Component
 * =========================================
 * Architecture:
 *  1. Camera auto-starts when component mounts (no manual button needed to begin)
 *  2. On each frame: calls /pose-extract → gets keypoints + raw_features
 *  3. raw_features → browser-side ONNX.js fall classifier (no cold start!)
 *  4. Temporal smoothing (weighted voting) → alert on confirmed fall
 *  5. If ONNX not available, falls back to server-side /inference
 *
 * Key UX changes:
 *  - Camera begins immediately; "Start Feed" toggle only pauses/resumes AI
 *  - "Waking up" overlay only for pose-extract server, not classifier
 *  - ONNX model loading indicator shown once on init
 *  - Faster capture interval: 1500ms (was 2000ms)
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { useWebcam } from '../hooks/useWebcam'
import { extractPose, sendFrame, logIncident } from '../services/api'
import {
  initOnnxModel,
  runFallClassifier,
  getOnnxStatus,
} from '../services/onnxInference'
import { Activity, VideoOff, Video, Cpu, AlertTriangle, WifiOff, Mic, MicOff } from 'lucide-react'
import { useVoiceDetection } from '../hooks/useVoiceDetection'

interface WebcamFeedProps {
  onFrameProcessed?: (result: any) => void
}

const POSE_CONNECTIONS = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6],
  [5, 7], [7, 9],
  [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
]

// Weighted temporal smoothing — more recent = more weight
const HISTORY_SIZE = 7
const FRAME_WEIGHTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] // oldest → newest
const FALL_WEIGHT_THRESHOLD = 5.5 // sum of weights needed to confirm fall

/**
 * Geometric rule-based fall detection from YOLOv8 17-keypoint output.
 * Works without ONNX — pure geometry on pixel coordinates.
 * Keypoints are in RAW PIXEL coordinates (not normalized).
 * Keypoint indices: 0=nose,5=l.shoulder,6=r.shoulder,11=l.hip,12=r.hip,
 *   13=l.knee,14=r.knee,15=l.ankle,16=r.ankle
 */
function detectFallGeometric(keypoints: number[][]): { detected: boolean; confidence: number; inference_ms: number; mode: 'onnx_browser' | 'server_fallback' } {
  const t0 = performance.now()
  const kp = keypoints

  // Get valid (non-zero) points — pixel coords so threshold > 2
  const get = (i: number) => (kp[i] && kp[i][0] > 2 && kp[i][1] > 2) ? kp[i] : null

  const lShoulder = get(5), rShoulder = get(6)
  const lHip = get(11), rHip = get(12)
  const lKnee = get(13), rKnee = get(14)
  const lAnkle = get(15), rAnkle = get(16)
  const nose = get(0)

  if (!lShoulder && !rShoulder) {
    console.log('[GEOMETRIC] Not enough keypoints to classify')
    return { detected: false, confidence: 0, inference_ms: Math.round(performance.now() - t0), mode: 'server_fallback' }
  }

  // Use available shoulders/hips
  const sh = lShoulder && rShoulder
    ? [(lShoulder[0] + rShoulder[0]) / 2, (lShoulder[1] + rShoulder[1]) / 2]
    : (lShoulder || rShoulder)!

  const hp = lHip && rHip
    ? [(lHip[0] + rHip[0]) / 2, (lHip[1] + rHip[1]) / 2]
    : lHip || rHip

  if (!hp) {
    console.log('[GEOMETRIC] No hip keypoints')
    return { detected: false, confidence: 0, inference_ms: Math.round(performance.now() - t0), mode: 'server_fallback' }
  }

  // Torso vector: shoulder → hip (in pixels)
  const torsoVecX = hp[0] - sh[0]
  const torsoVecY = hp[1] - sh[1]
  const torsoLen = Math.sqrt(torsoVecX ** 2 + torsoVecY ** 2)

  if (torsoLen < 10) {
    console.log('[GEOMETRIC] Torso too small:', torsoLen)
    return { detected: false, confidence: 0, inference_ms: Math.round(performance.now() - t0), mode: 'server_fallback' }
  }

  // Torso angle from vertical: 0° = upright, 90° = horizontal
  // atan2(|horizontal|, |vertical|) gives angle from vertical axis
  const torsoAngle = Math.atan2(Math.abs(torsoVecX), Math.abs(torsoVecY)) * (180 / Math.PI)

  // Shoulder width (horizontal span)
  const shoulderWidth = lShoulder && rShoulder ? Math.abs(lShoulder[0] - rShoulder[0]) : 0
  const bodyAspect = shoulderWidth / (torsoLen + 1)

  // Hip vertical position relative to knees/ankles
  let hipLow = false
  const kn = lKnee && rKnee ? [(lKnee[0] + rKnee[0]) / 2, (lKnee[1] + rKnee[1]) / 2]
    : lKnee || rKnee
  if (kn) {
    // In image coords, Y increases downward. Hip ABOVE knee means hp[1] < kn[1]
    // When person falls, hip[1] ≈ knee[1] or even hip[1] > knee[1]
    hipLow = hp[1] >= kn[1] * 0.85
  }

  // Nose below hip (person lying down)
  const noseBelowHip = nose ? nose[1] > hp[1] * 0.95 : false

  // Score calculation
  // bodyAspect = shoulderWidth / torsoLength
  // When standing: bodyAspect < 0.5 (torso longer than wide)
  // When lying flat: bodyAspect > 1.5 (shoulders wider than torso vector)
  let score = 0

  // Torso angle contribution
  if (torsoAngle > 65) score += 0.55
  else if (torsoAngle > 50) score += 0.35
  else if (torsoAngle > 35) score += 0.15

  // Body aspect ratio (horizontal body = fallen)
  if (bodyAspect > 2.0) score += 0.5       // extremely horizontal — strong fall
  else if (bodyAspect > 1.5) score += 0.35  // very horizontal
  else if (bodyAspect > 1.0) score += 0.2   // moderately horizontal
  else if (bodyAspect > 0.7) score += 0.1

  // Additional signals
  if (hipLow) score += 0.15
  if (noseBelowHip) score += 0.1

  const confidence = Math.min(score, 0.95)
  // Detect if: high body aspect alone OR torso angle + some other signal
  const detected = (bodyAspect > 1.5 && score >= 0.4) || (torsoAngle > 50 && score >= 0.45)

  console.log(`[GEOMETRIC] torsoAngle=${torsoAngle.toFixed(1)}° bodyAspect=${bodyAspect.toFixed(2)} hipLow=${hipLow} noseBelowHip=${noseBelowHip} → score=${score.toFixed(2)} detected=${detected}`)

  return {
    detected,
    confidence,
    inference_ms: Math.round(performance.now() - t0),
    mode: 'server_fallback'
  }
}

export function WebcamFeed({ onFrameProcessed }: WebcamFeedProps) {
  const { videoRef, canvasRef, isActive, setIsActive, captureFrame, cameraReady, cameraError } =
    useWebcam({ autoStart: true })

  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const fallHistoryRef = useRef<number[]>([]) // 0 or 1 per frame

  const [onnxReady, setOnnxReady] = useState(false)
  const [onnxLoading, setOnnxLoading] = useState(true)
  const [inferenceMode, setInferenceMode] = useState<'onnx_browser' | 'server_fallback' | null>(null)

  const [isProcessing, setIsProcessing] = useState(false)
  const [isServerWakingUp, setIsServerWakingUp] = useState(false)
  const [status, setStatus] = useState('Initializing...')
  const [fps, setFps] = useState<number | null>(null)
  const [inferenceTime, setInferenceTime] = useState<number | null>(null)

  const [keypoints, setKeypoints] = useState<number[][] | null>(null)
  const [box, setBox] = useState<number[] | null>(null)
  const [isFallDetected, setIsFallDetected] = useState(false)
  const [confidence, setConfidence] = useState(0)

  // ---- Voice Detection Integration ---------------------------------------
  const { isListening, error: voiceError, lastDetected } = useVoiceDetection({
    enabled: isActive && cameraReady,
    onDetect: (result) => {
      // Log incident immediately
      logIncident({
        type: 'voice_distress',
        confidence: result.confidence,
        inference_ms: 0,
        source: 'browser_speech_api'
      })
    }
  })

  // ---- ONNX Initialization (runs once on mount) ---------------------------

  useEffect(() => {
    let cancelled = false
    setOnnxLoading(true)
    setStatus('Loading on-device AI model...')

    initOnnxModel().then((ok) => {
      if (cancelled) return
      setOnnxReady(ok)
      setOnnxLoading(false)
      if (ok) {
        setStatus('AI Ready — Scanning...')
        setInferenceMode('onnx_browser')
      } else {
        setStatus('Using server inference (fallback)')
        setInferenceMode('server_fallback')
      }
    })
    return () => { cancelled = true }
  }, [])

  // ---- Canvas Overlay Drawing --------------------------------------------

  const clearOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current
    if (!canvas) return
    canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
  }, [])

  const drawOverlay = useCallback(
    (kpts: number[][] | null, boundingBox: number[] | null, isFall: boolean, conf: number, hazards: any[] = []) => {
      const canvas = overlayCanvasRef.current
      const video = videoRef.current
      if (!canvas || !video) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const rect = video.getBoundingClientRect()
      canvas.width = rect.width
      canvas.height = rect.height
      const { width, height } = canvas
      ctx.clearRect(0, 0, width, height)

      // HUD reticle corners
      const hudColor = isFall ? 'rgba(239,68,68,0.9)' : (hazards.length > 0 ? 'rgba(249,115,22,0.9)' : 'rgba(99,102,241,0.5)')
      ctx.strokeStyle = hudColor
      ctx.lineWidth = 2
      const len = 20, pad = 12
      const corners = [
        [pad, pad], [width - pad, pad], [pad, height - pad], [width - pad, height - pad],
      ]
      const dirs = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
      corners.forEach(([x, y], i) => {
        const [dx, dy] = dirs[i]
        ctx.beginPath()
        ctx.moveTo(x, y + dy * len); ctx.lineTo(x, y); ctx.lineTo(x + dx * len, y)
        ctx.stroke()
      })

      // Draw hazards (Fire/Smoke)
      hazards.forEach(hazard => {
        const [xmin, ymin, xmax, ymax] = hazard.box // Server returns normalized coords
        const bx = xmin * width, by = ymin * height
        const bw = (xmax - xmin) * width, bh = (ymax - ymin) * height
        const hazardColor = hazard.type === 'fire' ? '#ef4444' : '#f97316' // Red for fire, Orange for smoke
        
        ctx.strokeStyle = hazardColor
        ctx.lineWidth = 3
        ctx.shadowColor = hazardColor
        ctx.shadowBlur = 15
        ctx.strokeRect(bx, by, bw, bh)
        ctx.shadowBlur = 0

        // Hazard Label
        const label = `🔥 ${hazard.type.toUpperCase()} ${(hazard.confidence * 100).toFixed(0)}%`
        ctx.fillStyle = hazardColor
        ctx.font = 'bold 10px monospace'
        const labelW = Math.max(ctx.measureText(label).width + 8, 90)
        ctx.fillRect(bx, by - 18, labelW, 18)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, bx + 4, by - 5)
      })

      const color = isFall ? '#ef4444' : '#10b981'

      // Bounding box for person
      if (boundingBox) {
        const [ymin, xmin, ymax, xmax] = boundingBox // YOLO returns [ymin, xmin, ymax, xmax]
        const bx = xmin * width, by = ymin * height
        const bw = (xmax - xmin) * width, bh = (ymax - ymin) * height
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.shadowColor = color
        ctx.shadowBlur = 10
        ctx.strokeRect(bx, by, bw, bh)
        ctx.shadowBlur = 0

        // Label
        const label = isFall
          ? `⚠ FALL DETECTED ${(conf * 100).toFixed(0)}%`
          : `TRACKING ${(conf * 100).toFixed(0)}%`
        ctx.fillStyle = color
        ctx.font = 'bold 9px monospace'
        const labelW = Math.max(ctx.measureText(label).width + 8, 80)
        ctx.fillRect(bx, by - 16, labelW, 16)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, bx + 4, by - 5)
      } else if (isFall) {
        // If we don't have a bounding box but fall is detected (e.g. HuggingFace fallback)
        const label = `⚠ FALL DETECTED ${(conf * 100).toFixed(0)}% (NO BBOX)`
        ctx.fillStyle = color
        ctx.font = 'bold 12px monospace'
        const labelW = ctx.measureText(label).width + 16
        const bx = (width - labelW) / 2
        const by = height / 2
        ctx.fillRect(bx, by - 20, labelW, 20)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, bx + 8, by - 5)
      }

      if (!kpts) return

      // Skeleton connections
      ctx.strokeStyle = color
      ctx.lineWidth = 2.5
      POSE_CONNECTIONS.forEach(([i, j]) => {
        const kp1 = kpts[i], kp2 = kpts[j]
        if (kp1 && kp2 && kp1[0] > 0.01 && kp2[0] > 0.01) {
          ctx.beginPath()
          ctx.moveTo(kp1[0] * width, kp1[1] * height)
          ctx.lineTo(kp2[0] * width, kp2[1] * height)
          ctx.stroke()
        }
      })

      // Joint dots
      kpts.forEach((kp) => {
        if (kp && kp[0] > 0.01) {
          ctx.fillStyle = color
          ctx.beginPath()
          ctx.arc(kp[0] * width, kp[1] * height, 4.5, 0, Math.PI * 2)
          ctx.fill()
          ctx.strokeStyle = isFall ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.arc(kp[0] * width, kp[1] * height, 7.5, 0, Math.PI * 2)
          ctx.stroke()
        }
      })
    },
    [videoRef]
  )

  useEffect(() => {
    if (!isActive) return
    const handleResize = () => drawOverlay(keypoints, box, isFallDetected, confidence)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [isActive, keypoints, box, isFallDetected, confidence, drawOverlay])

  // ---- Weighted Temporal Smoothing ----------------------------------------

  const updateFallHistory = useCallback((detected: boolean): boolean => {
    const history = fallHistoryRef.current
    history.push(detected ? 1 : 0)
    if (history.length > HISTORY_SIZE) history.shift()

    const start = HISTORY_SIZE - history.length
    let weightedSum = 0
    history.forEach((val, i) => {
      weightedSum += val * FRAME_WEIGHTS[start + i]
    })
    return weightedSum >= FALL_WEIGHT_THRESHOLD
  }, [])

  // ---- Main Capture Loop --------------------------------------------------

  useEffect(() => {
    if (!isActive || !cameraReady) {
      if (!isActive) {
        setStatus('Monitoring Paused')
        setFps(null)
        setInferenceTime(null)
        setKeypoints(null)
        setBox(null)
        setIsFallDetected(false)
        setConfidence(0)
        fallHistoryRef.current = []
        clearOverlay()
      }
      return
    }

    setStatus(onnxReady ? 'Scanning (On-Device AI)...' : 'Scanning (Server AI)...')
    let lastFrameTime = Date.now()

    const captureInterval = setInterval(async () => {
      const frameBase64 = captureFrame()
      if (!frameBase64) return

      setIsProcessing(true)
      const wakingTimer = setTimeout(() => setIsServerWakingUp(true), 3000)

      const t0 = performance.now()

      try {
        // --- UNIFIED: Always use /pose-extract for keypoints + hazard detection ---
        // This guarantees skeleton is always drawn. Classification then tries ONNX,
        // and falls back to geometric rules if ONNX is unavailable.
        const poseResult = await extractPose(frameBase64)
        clearTimeout(wakingTimer)
        setIsServerWakingUp(false)

        if (poseResult.status === 'ok') {
          const hazards = poseResult.hazards || []
          let isFallSmoothed = false
          let fallConfidence = 0

          if (poseResult.raw_features) {
            let prediction = null

            // Try browser ONNX first (fastest)
            if (onnxReady) {
              prediction = await runFallClassifier(poseResult.raw_features)
            }

            // Geometric rule-based fallback: always works, no ONNX needed
            if (!prediction && poseResult.keypoints && poseResult.keypoints.length >= 17) {
              prediction = detectFallGeometric(poseResult.keypoints)
              console.log('[GEOMETRIC] result:', prediction)
            }

            if (prediction) {
              setInferenceMode(onnxReady ? 'onnx_browser' : 'server_fallback')
              fallConfidence = prediction.confidence
              setConfidence(fallConfidence)
              isFallSmoothed = updateFallHistory(prediction.detected)
              setIsFallDetected(isFallSmoothed)

              if (isFallSmoothed) {
                logIncident({
                  type: 'fall',
                  confidence: fallConfidence,
                  inference_ms: Math.round(performance.now() - t0),
                  source: onnxReady ? 'client_onnx' : 'geometric_rules',
                })
              }
            }
          } else {
            updateFallHistory(false)
            setIsFallDetected(false)
          }

          // Always render keypoints + hazards on canvas
          setKeypoints(poseResult.keypoints || null)
          setBox(poseResult.box || null)

          console.log('[POSE] keypoints received:', poseResult.keypoints?.length, '| person_detected:', poseResult.person_detected, '| isFall:', isFallSmoothed, '| conf:', fallConfidence.toFixed(2))

          if (poseResult.keypoints || hazards.length > 0) {
            // Keypoints from backend are RAW PIXEL COORDS. Normalize to [0,1] using video dimensions.
            let normalizedKpts: number[][] | null = null
            if (poseResult.keypoints && poseResult.keypoints.length > 0) {
              const vid = videoRef.current
              const vidW = vid?.videoWidth || 640
              const vidH = vid?.videoHeight || 480
              normalizedKpts = poseResult.keypoints.map(([x, y]: number[]) => [x / vidW, y / vidH])
              console.log('[POSE] Normalized kpt[5] (l.shoulder):', normalizedKpts[5])
            }
            drawOverlay(normalizedKpts, poseResult.box || null, isFallSmoothed, fallConfidence, hazards)
          } else {
            clearOverlay()
          }

          // Log fire/smoke hazards
          hazards.forEach((hazard: any) => {
            logIncident({
              type: hazard.type,
              confidence: hazard.confidence,
              inference_ms: Math.round(performance.now() - t0),
              source: 'server_yolo',
            })
          })

          onFrameProcessed?.({
            status: 'processed',
            detected: isFallSmoothed || hazards.length > 0,
            confidence: Math.max(fallConfidence, ...hazards.map((h: any) => h.confidence), 0),
            ai_result: {
              detected: isFallSmoothed,
              confidence: fallConfidence,
              type: isFallSmoothed ? 'fall' : (hazards.length > 0 ? hazards[0].type : 'normal'),
              inference_ms: Math.round(performance.now() - t0),
              keypoints: poseResult.keypoints,
              box: poseResult.box,
            },
          })
        } else {
          updateFallHistory(false)
          setIsFallDetected(false)
          setKeypoints(null); setBox(null); clearOverlay()
        }

        // FPS + status
        const now = Date.now()
        const elapsed = now - lastFrameTime
        setFps(Math.round(1000 / elapsed))
        lastFrameTime = now
        setInferenceTime(Math.round(performance.now() - t0))
        setStatus(onnxReady ? 'On-Device AI Active' : 'Server AI + Geometric Rules Active')

      } catch (err) {
        clearTimeout(wakingTimer)
        setIsServerWakingUp(false)
        setStatus('Connection error — retrying...')
        setKeypoints(null); setBox(null); clearOverlay()
        console.error('[WebcamFeed] Frame error:', err)
      } finally {
        setIsProcessing(false)
      }
    }, 1500)  // 1.5s interval (was 2s)

    return () => clearInterval(captureInterval)
  }, [
    isActive, cameraReady, onnxReady,
    captureFrame, onFrameProcessed,
    clearOverlay, drawOverlay, updateFallHistory,
  ])

  // ---- Render -------------------------------------------------------------

  return (
    <div
      className={`w-full h-full bg-slate-950/60 backdrop-blur-md rounded-2xl border transition-all duration-500 overflow-hidden shadow-2xl flex flex-col justify-between ${
        isFallDetected ? 'border-rose-500 shadow-rose-950/20' : 'border-slate-800/80 shadow-slate-950/50'
      }`}
    >
      {/* HUD Header */}
      <div className="absolute top-0 left-0 w-full bg-black/60 backdrop-blur-md px-4 py-2 flex items-center justify-between z-20 border-b border-slate-800/40">
        <div className="flex items-center gap-2">
          {isFallDetected ? (
            <AlertTriangle className="w-4 h-4 text-rose-500 animate-pulse" />
          ) : (
            <Activity className="w-4 h-4 text-secondary animate-pulse" />
          )}
          <span className={`text-[10px] uppercase tracking-wider font-semibold ${isFallDetected ? 'text-rose-300 font-bold' : 'text-slate-300'}`}>
            {isFallDetected ? 'CRITICAL — FALL DETECTED' : 'AI Sensor Telemetry'}
          </span>
        </div>
        <div className="flex items-center gap-4 text-[9px] text-slate-400 font-mono">
          {/* Voice Detection status */}
          {isActive && cameraReady && (
            <span className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[8px] font-bold uppercase tracking-wider ${
              lastDetected ? 'bg-rose-950/60 text-rose-400 border border-rose-800/50 animate-pulse' : 
              isListening ? 'bg-indigo-950/60 text-indigo-400 border border-indigo-800/50' : 'bg-slate-800/60 text-slate-500'
            }`} title={voiceError || (isListening ? 'Listening for keywords...' : 'Voice disabled')}>
              {isListening ? <Mic className="w-2.5 h-2.5" /> : <MicOff className="w-2.5 h-2.5" />}
              {lastDetected ? `HEARD: ${lastDetected}` : isListening ? 'VOICE ON' : 'VOICE OFF'}
            </span>
          )}

          {/* Inference mode badge */}
          {inferenceMode && (
            <span className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[8px] font-bold uppercase tracking-wider ${
              inferenceMode === 'onnx_browser'
                ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/50'
                : 'bg-amber-950/60 text-amber-400 border border-amber-800/50'
            }`}>
              <Cpu className="w-2.5 h-2.5" />
              {inferenceMode === 'onnx_browser' ? 'On-Device' : 'Server'}
            </span>
          )}
          {fps !== null && <span>FPS: {fps}</span>}
          {inferenceTime !== null && <span>Latency: {inferenceTime}ms</span>}
        </div>
      </div>

      {/* Camera Viewport */}
      <div className="relative flex-grow bg-slate-950 flex items-center justify-center overflow-hidden min-h-[300px]">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover transition-opacity duration-500 ${
            isActive && cameraReady ? 'opacity-75' : 'opacity-0 h-0 w-0'
          }`}
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* Overlay Canvas */}
        {isActive && cameraReady && (
          <canvas
            ref={overlayCanvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none z-10"
          />
        )}

        {/* Scanline effect */}
        {isActive && cameraReady && (
          <div className={`absolute inset-0 pointer-events-none scanline-effect border border-slate-800/20 ${isFallDetected ? 'bg-rose-950/5' : ''}`} />
        )}

        {/* Camera Error */}
        {cameraError && (
          <div className="flex flex-col items-center justify-center p-8 text-center">
            <WifiOff className="w-10 h-10 text-rose-500 mb-3" />
            <h3 className="text-sm font-semibold text-rose-300 font-sans">Camera Access Denied</h3>
            <p className="text-xs text-slate-400 max-w-xs mt-1 font-sans">{cameraError}</p>
            <p className="text-xs text-slate-500 mt-2">Allow camera access in your browser settings, then refresh.</p>
          </div>
        )}

        {/* ONNX Model Loading (once) */}
        {onnxLoading && !cameraError && (
          <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center z-20 text-center p-6">
            <div className="relative mb-4">
              <div className="w-14 h-14 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin" />
              <Cpu className="absolute inset-0 m-auto w-6 h-6 text-indigo-400" />
            </div>
            <h3 className="text-sm font-semibold text-indigo-300 font-sans">Loading On-Device AI</h3>
            <p className="text-xs text-slate-400 max-w-xs mt-1.5 font-sans">
              Initializing local fall detection model. This only happens once.
            </p>
          </div>
        )}

        {/* Server Waking Up (only for pose extraction) */}
        {isServerWakingUp && !onnxLoading && (
          <div className="absolute inset-0 bg-slate-950/75 backdrop-blur-sm flex flex-col items-center justify-center z-20 text-center p-6">
            <div className="w-12 h-12 rounded-full border-2 border-amber-500 border-t-transparent animate-spin mb-4" />
            <h3 className="text-sm font-semibold text-amber-400 animate-pulse font-sans">Waking up Pose Server...</h3>
            <p className="text-xs text-slate-400 max-w-xs mt-1.5 font-sans">
              The pose extraction server is starting. Classification will continue locally.
            </p>
          </div>
        )}

        {/* Standby (paused) */}
        {!isActive && !cameraError && (
          <div className="flex flex-col items-center justify-center p-8 text-center text-slate-500">
            <div className="w-16 h-16 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center mb-4">
              <VideoOff className="w-6 h-6 text-slate-400" />
            </div>
            <h3 className="text-sm font-semibold text-slate-300 font-sans">Monitoring Paused</h3>
            <p className="text-xs text-slate-500 max-w-xs mt-1 font-sans">
              Resume the feed to restart AI safety monitoring.
            </p>
          </div>
        )}

        {/* LIVE indicator */}
        {isActive && cameraReady && (
          <div className={`absolute top-4 left-4 backdrop-blur-md px-3 py-1.5 rounded-lg border text-white flex items-center gap-2 z-25 transition-colors duration-500 ${
            isFallDetected ? 'bg-rose-950/85 border-rose-800/60' : 'bg-slate-950/85 border-slate-800'
          }`}>
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isFallDetected ? 'bg-rose-400' : 'bg-emerald-400'}`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isFallDetected ? 'bg-rose-500' : 'bg-emerald-500'}`} />
            </span>
            <span className={`text-[9px] uppercase font-mono tracking-widest font-bold ${isFallDetected ? 'text-rose-400' : 'text-emerald-400'}`}>
              {isFallDetected ? 'ALERT: FALL' : 'LIVE'}
            </span>
          </div>
        )}

        {/* Confidence meter */}
        {isActive && cameraReady && confidence > 0 && (
          <div className="absolute top-4 right-4 backdrop-blur-md px-2 py-1.5 rounded-lg border border-slate-700 bg-slate-950/85 z-25">
            <p className="text-[8px] font-mono text-slate-400 uppercase tracking-widest mb-1">Confidence</p>
            <div className="w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${confidence >= 0.65 ? 'bg-rose-500' : 'bg-emerald-500'}`}
                style={{ width: `${Math.min(confidence * 100, 100)}%` }}
              />
            </div>
            <p className={`text-[9px] font-mono mt-0.5 text-right ${confidence >= 0.65 ? 'text-rose-400' : 'text-emerald-400'}`}>
              {(confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}

        {/* Critical Fall Alert Overlay */}
        {isActive && isFallDetected && (
          <div className="absolute inset-0 border-4 border-rose-500 animate-pulse pointer-events-none z-30 flex items-center justify-center bg-rose-500/5">
            <div className="bg-rose-600/90 backdrop-blur-md text-white font-black text-xs tracking-widest px-4 py-3 rounded-2xl flex items-center gap-2.5 border border-rose-400 shadow-2xl animate-bounce pointer-events-auto font-sans">
              <AlertTriangle className="w-4 h-4" />
              <span>WARNING: FALL DETECTED!</span>
            </div>
          </div>
        )}

        {/* Processing indicator */}
        {isProcessing && !isServerWakingUp && (
          <div className="absolute bottom-4 right-4 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 flex items-center gap-2 text-indigo-400 z-20">
            <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse" />
            <span className="text-[9px] font-mono font-semibold uppercase tracking-wider">Evaluating...</span>
          </div>
        )}
      </div>

      {/* Control Panel Footer */}
      <div className={`p-3 border-t flex justify-between items-center gap-4 transition-colors duration-500 ${
        isFallDetected ? 'bg-rose-950/10 border-rose-950' : 'bg-slate-900/30 border-slate-800/40'
      }`}>
        <span className="text-xs text-slate-400 font-mono">
          Status:{' '}
          <span className={`${isFallDetected ? 'text-rose-400 font-bold' : 'text-slate-200 font-semibold'}`}>
            {status}
          </span>
        </span>
        <button
          onClick={() => setIsActive(!isActive)}
          className={`flex items-center gap-1.5 px-4 py-2 rounded-xl font-bold text-[10px] tracking-wider uppercase transition-all duration-300 cursor-pointer ${
            isActive
              ? 'bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white hover:translate-y-[-1px]'
          }`}
        >
          {isActive ? (
            <><VideoOff className="w-3.5 h-3.5" />Pause Feed</>
          ) : (
            <><Video className="w-3.5 h-3.5" />Resume Feed</>
          )}
        </button>
      </div>
    </div>
  )
}
