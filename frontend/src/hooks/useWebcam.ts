import { useEffect, useRef, useState, useCallback } from 'react'

export interface UseWebcamOptions {
  /** If true, automatically requests camera access on mount */
  autoStart?: boolean
}

export interface UseWebcamReturn {
  videoRef: React.RefObject<HTMLVideoElement>
  canvasRef: React.RefObject<HTMLCanvasElement>
  isActive: boolean
  setIsActive: React.Dispatch<React.SetStateAction<boolean>>
  captureFrame: () => string | null
  cameraReady: boolean
  cameraError: string | null
}

/**
 * useWebcam — Manages webcam access and frame capture.
 *
 * Changes from original:
 * - Added `autoStart` option (default: true) to request camera on mount
 * - Added `cameraReady` state — true when video stream has started playing
 * - Added `cameraError` state — captures permission denied / hardware errors
 * - Graceful cleanup on unmount
 */
export function useWebcam(options: UseWebcamOptions = {}): UseWebcamReturn {
  const { autoStart = true } = options

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const [isActive, setIsActive] = useState(autoStart)
  const [cameraReady, setCameraReady] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)

  useEffect(() => {
    if (!isActive) {
      // Stop camera stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
      setCameraReady(false)
      return
    }

    let cancelled = false

    const startWebcam = async () => {
      setCameraError(null)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
          },
        })

        if (cancelled) {
          stream.getTracks().forEach(t => t.stop())
          return
        }

        streamRef.current = stream

        if (videoRef.current) {
          videoRef.current.srcObject = stream

          // Wait for video to actually start playing before signalling ready
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().then(() => {
              if (!cancelled) setCameraReady(true)
            })
          }
        }
      } catch (err) {
        if (cancelled) return
        const msg = err instanceof Error ? err.message : 'Unknown camera error'
        console.error('[useWebcam] Error accessing camera:', err)
        setCameraError(msg)
        setIsActive(false)
      }
    }

    startWebcam()

    return () => {
      cancelled = true
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
      setCameraReady(false)
    }
  }, [isActive])

  /**
   * Capture the current video frame as a base64-encoded JPEG string (no prefix).
   * Returns null if camera is not ready or canvas fails.
   */
  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !cameraReady) return null
    if (video.videoWidth === 0 || video.videoHeight === 0) return null

    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    // Return base64 without the data URI prefix
    return canvas.toDataURL('image/jpeg', 0.7).split(',')[1]
  }, [cameraReady])

  return {
    videoRef,
    canvasRef,
    isActive,
    setIsActive,
    captureFrame,
    cameraReady,
    cameraError,
  }
}
