from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import base64
import pickle
from collections import deque

from app.core.config import settings
from app.services.notification_service import send_incident_email
from supabase import create_client, Client

# Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Resolve model and scaler paths relative to the project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "safewatch_model_cpu.onnx"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "safewatch_model_cpu.onnx"),
    "models/safewatch_model_cpu.onnx",
    os.path.join(BASE_DIR, "models", "safewatch_model_cpu_float32.onnx"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "safewatch_model_cpu_float32.onnx"),
    "models/safewatch_model_cpu_float32.onnx"
]

SCALER_PATHS = [
    os.path.join(BASE_DIR, "models", "feature_scaler.pkl"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_scaler.pkl"),
    "models/feature_scaler.pkl"
]

YOLO_POSE_PATHS = [
    os.path.join(BASE_DIR, "models", "yolov8n-pose.pt"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.pt"),
    "yolov8n-pose.pt"
]

YOLO_FIRE_PATHS = [
    os.path.join(BASE_DIR, "models", "fire_smoke.pt"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fire_smoke.pt"),
    "models/fire_smoke.pt"
]

# Startup Model Loading
yolo_pose_model = None
for path in YOLO_POSE_PATHS:
    if os.path.exists(path):
        try:
            from ultralytics import YOLO
            yolo_pose_model = YOLO(path)
            print(f"YOLOv8 Pose model loaded from {path}")
            break
        except Exception as e:
            print(f"Warning: Failed to load YOLOv8 Pose model from {path}: {e}")

if yolo_pose_model is None:
    try:
        from ultralytics import YOLO
        yolo_pose_model = YOLO("yolov8n-pose.pt")
        print("YOLOv8 Pose model loaded from default/downloaded path")
    except Exception as e:
        print(f"Warning: Fallback YOLOv8 Pose model download/load failed: {e}")

yolo_fire_model = None
for path in YOLO_FIRE_PATHS:
    if os.path.exists(path):
        try:
            from ultralytics import YOLO
            yolo_fire_model = YOLO(path)
            print(f"YOLO Fire/Smoke model loaded from {path}")
            break
        except Exception as e:
            print(f"Warning: Failed to load YOLO Fire model from {path}: {e}")

if yolo_fire_model is None:
    print("Warning: YOLO Fire/Smoke model could not be loaded. Fire detection will be disabled.")

fall_model = None
fall_model_input_name = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            fall_model = ort.InferenceSession(path, sess_options=opts)
            fall_model_input_name = fall_model.get_inputs()[0].name
            print(f"ONNX CPU model loaded from {path}")
            break
        except Exception as e:
            print(f"Warning: Failed to load ONNX model from {path}: {e}")

if fall_model is None:
    print("Warning: ONNX model (safewatch_model_cpu.onnx) could not be loaded.")

scaler = None
for path in SCALER_PATHS:
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
            print(f"Feature scaler loaded successfully from {path}")
            break
        except Exception as e:
            print(f"Warning: Failed to load feature scaler from {path}: {e}")

if scaler is None:
    print("Warning: feature_scaler.pkl could not be loaded.")

app = FastAPI(title=settings.PROJECT_NAME)
app.state.scaler = scaler
app.state.fall_model = fall_model
app.state.fall_model_input_name = fall_model_input_name
app.state.yolo_pose_model = yolo_pose_model
app.state.yolo_fire_model = yolo_fire_model
app.state.fall_history = deque(maxlen=5)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FrameInput(BaseModel):
    image: str  # Base64 encoded image

async def process_frame_pipeline(frame: FrameInput):
    start_time = time.time()
    
    # 1. Check if Hugging Face Inference API is configured and should be used
    is_hf_configured = (
        settings.HF_API_URL 
        and "placeholder" not in settings.HF_API_URL 
        and settings.HF_API_URL != "https://api-inference.huggingface.co/models/"
    )
    
    if is_hf_configured:
        try:
            from app.services.ai_service import query_hf_inference
            ai_result = query_hf_inference(frame.image)
            
            if ai_result:
                is_fall = ai_result.get("detected", False)
                prob_fall = ai_result.get("confidence", 0.0)
                
                # Apply temporal smoothing using history
                app.state.fall_history.append(1 if is_fall else 0)
                is_fall_smoothed = sum(app.state.fall_history) >= 3
                
                if is_fall_smoothed:
                    # Store in Supabase
                    incident_data = {
                        "type": ai_result.get("type", "fall"),
                        "confidence": prob_fall,
                        "inference_ms": ai_result.get("inference_ms", int((time.time() - start_time) * 1000)),
                        "created_at": "now()"
                    }
                    try:
                        supabase.table("incidents").insert(incident_data).execute()
                        print("Success: Incident logged to Supabase.")
                    except Exception as e:
                        print(f"Supabase logging failed: {e}")
                        
                    # Send Email Notification
                    try:
                        send_incident_email(ai_result.get("type", "fall"), prob_fall)
                        print("Success: Email notification triggered.")
                    except Exception as e:
                        print(f"Email notification failed: {e}")
                
                inference_ms = int((time.time() - start_time) * 1000)
                return {
                    "status": "processed",
                    "ai_result": {
                        "detected": bool(is_fall_smoothed),
                        "confidence": float(prob_fall),
                        "type": ai_result.get("type", "fall"),
                        "inference_ms": inference_ms,
                        "keypoints": None,
                        "box": None
                    },
                    "detected": bool(is_fall_smoothed),
                    "confidence": float(prob_fall)
                }
        except Exception as e:
            print(f"Warning: Hugging Face inference failed, falling back to local: {e}")
            
    # 2. Local Fallback (YOLOv8 + ONNX)
    try:
        import numpy as np
        import cv2
        try:
            from app.services.advanced_feature_engineering import AdvancedFeatureEngineer
        except ImportError:
            from src.advanced_feature_engineering import AdvancedFeatureEngineer
    except ImportError as e:
        print(f"Warning: Local ML dependencies missing, returning mock fallback: {e}")
        return {
            "status": "processed",
            "ai_result": {
                "detected": False,
                "confidence": 0.0,
                "type": "normal",
                "inference_ms": 10
            },
            "detected": False,
            "confidence": 0.0,
            "reason": f"missing_dependencies_and_no_api: {str(e)}"
        }
        
    # Decode base64 frame image
    image_base64 = frame.image
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    try:
        image_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": f"invalid_image_format: {str(e)}",
            "ai_result": {
                "detected": False,
                "confidence": 0.0,
                "reason": f"invalid_image_format: {str(e)}",
                "type": "normal",
                "keypoints": None,
                "box": None
            }
        }
        
    if app.state.yolo_pose_model is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "yolo_model_not_loaded",
            "ai_result": {
                "detected": False,
                "confidence": 0.0,
                "reason": "yolo_model_not_loaded",
                "type": "normal",
                "keypoints": None,
                "box": None
            }
        }
        
    kp_person_list = None
    box = None
    results = app.state.yolo_pose_model(img, verbose=False)
    
    landmarks = None
    if results and len(results) > 0 and results[0].keypoints is not None:
        xy = results[0].keypoints.xy
        if xy is not None and len(xy) > 0:
            kp_person = xy[0].cpu().numpy() # shape (17, 2) in raw pixels
            kp_person_list = kp_person.tolist()
            if kp_person.shape[0] >= 17:
                # Extract shoulder, hip, knee, ankle indices (5, 6, 11, 12, 13, 14, 15, 16)
                indices = [5, 6, 11, 12, 13, 14, 15, 16]
                selected_kpts = kp_person[indices] # shape (8, 2)
                landmarks = selected_kpts.flatten() # shape (16,) in raw pixels
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            box = results[0].boxes.xyxyn[0].cpu().numpy().tolist()

    if landmarks is None:
        landmarks = np.zeros(16, dtype=np.float32)
        
    # b. VALIDATE landmarks - if sum is too small (e.g. 0), return error
    if np.sum(np.abs(landmarks)) < 1.0:
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "no_pose_detected",
            "ai_result": {
                "detected": False,
                "confidence": 0.0,
                "reason": "no_pose_detected",
                "type": "normal",
                "keypoints": None,
                "box": None
            }
        }
        
    # c. Call AdvancedFeatureEngineer to get biological normalized features (34 total)
    features_dict = AdvancedFeatureEngineer.extract_geometric_features(landmarks)
    total_features = np.array(list(features_dict.values()), dtype=np.float32) # shape (34,)
    
    # e. Apply scaler transform before passing to model
    if app.state.scaler is None or app.state.fall_model is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "ml_models_not_loaded",
            "ai_result": {
                "detected": False,
                "confidence": 0.0,
                "reason": "ml_models_not_loaded",
                "type": "normal",
                "keypoints": None,
                "box": None
            }
        }
        
    features_scaled = app.state.scaler.transform(total_features.reshape(1, -1)).astype(np.float32)
    
    # Run model prediction using ONNX Runtime
    try:
        input_name = getattr(app.state, "fall_model_input_name", None)
        if not input_name:
            input_name = app.state.fall_model.get_inputs()[0].name
        
        outputs = app.state.fall_model.run(None, {input_name: features_scaled})
        prob_fall = float(outputs[0][0][0])
    except Exception as e:
        print(f"Error running ONNX model inference: {e}")
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": f"inference_error: {str(e)}",
            "ai_result": {
                "detected": False,
                "confidence": 0.0,
                "reason": f"inference_error: {str(e)}",
                "type": "normal",
                "keypoints": None,
                "box": None
            }
        }
    
    # f. Use CONFIDENCE_THRESHOLD env var (default 0.65) for classification
    conf_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", settings.CONFIDENCE_THRESHOLD or 0.65))
    is_fall_current = prob_fall >= conf_threshold
    
    # 3. Temporal smoothing using deque
    app.state.fall_history.append(1 if is_fall_current else 0)
    is_fall_smoothed = sum(app.state.fall_history) >= 3
    
    # Trigger alerts (incident logging to Supabase + email notification) on smoothed fall
    if is_fall_smoothed:
        # 2. Store in Supabase
        incident_data = {
            "type": "fall",
            "confidence": prob_fall,
            "inference_ms": int((time.time() - start_time) * 1000),
            "created_at": "now()"
        }
        try:
            supabase.table("incidents").insert(incident_data).execute()
            print("Success: Incident logged to Supabase.")
        except Exception as e:
            print(f"Supabase logging failed: {e}")
            
        # 3. Send Email Notification (with cooldown)
        try:
            send_incident_email("fall", prob_fall)
            print("Success: Email notification triggered.")
        except Exception as e:
            print(f"Email notification failed: {e}")
            
    inference_ms = int((time.time() - start_time) * 1000)
    ai_result = {
        "detected": bool(is_fall_smoothed),
        "confidence": float(prob_fall),
        "type": "fall" if is_fall_smoothed else "normal",
        "inference_ms": inference_ms,
        "keypoints": kp_person_list,
        "box": box
    }
    
    return {
        "status": "processed",
        "ai_result": ai_result,
        "detected": bool(is_fall_smoothed),
        "confidence": float(prob_fall)
    }


@app.post("/inference")
async def inference(frame: FrameInput):
    return await process_frame_pipeline(frame)

@app.post("/detect")
async def detect(frame: FrameInput):
    return await process_frame_pipeline(frame)

@app.post("/pose-extract")
async def pose_extract(frame: FrameInput):
    """
    Lightweight endpoint: extract YOLOv8 pose keypoints only.
    The fall classification is done client-side via ONNX.js.
    Returns the 17 keypoints + bounding box. No fall logic here.
    """
    try:
        import numpy as np
        import cv2
    except ImportError as e:
        return {"status": "error", "keypoints": None, "box": None, "reason": str(e)}

    # Decode base64 image
    image_base64 = frame.image
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    try:
        image_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        return {"status": "error", "keypoints": None, "box": None, "reason": f"invalid_image: {e}"}

    if app.state.yolo_pose_model is None:
        return {"status": "error", "keypoints": None, "box": None, "reason": "yolo_not_loaded"}

    results = app.state.yolo_pose_model(img, verbose=False)

    keypoints_list = None
    box = None
    raw_features = None  # 16-float landmark array for ONNX client-side

    if results and len(results) > 0 and results[0].keypoints is not None:
        xy = results[0].keypoints.xy
        if xy is not None and len(xy) > 0:
            kp_person = xy[0].cpu().numpy()  # shape (17, 2) raw pixels
            keypoints_list = kp_person.tolist()

            if kp_person.shape[0] >= 17:
                # Extract the 8 key joints used by the fall classifier
                indices = [5, 6, 11, 12, 13, 14, 15, 16]
                selected_kpts = kp_person[indices]  # shape (8, 2)
                raw_features = selected_kpts.flatten().tolist()  # 16 floats (pixels)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            box = results[0].boxes.xyxyn[0].cpu().numpy().tolist()

    # --- FIRE AND SMOKE DETECTION ---
    hazards = []
    if app.state.yolo_fire_model is not None:
        fire_results = app.state.yolo_fire_model(img, verbose=False)
        if fire_results and len(fire_results) > 0 and fire_results[0].boxes is not None:
            for fire_box in fire_results[0].boxes:
                cls_id = int(fire_box.cls[0])
                conf = float(fire_box.conf[0])
                label = app.state.yolo_fire_model.names[cls_id].lower()
                
                # Minimum confidence threshold for fire/smoke
                if conf > 0.65 and (label == 'fire' or label == 'smoke'):
                    bbox_n = fire_box.xyxyn[0].cpu().numpy().tolist() # Normalized box
                    hazards.append({
                        "type": label,
                        "confidence": conf,
                        "box": bbox_n
                    })

    return {
        "status": "ok",
        "keypoints": keypoints_list,     # Full 17 kpts for drawing skeleton
        "raw_features": raw_features,    # 16 floats for ONNX fall classifier
        "box": box,
        "person_detected": keypoints_list is not None,
        "hazards": hazards               # Array of detected fire/smoke
    }

@app.post("/log-incident")
async def log_incident(data: dict):
    """
    Endpoint for client-side fall detection to log confirmed incidents.
    Called by frontend after ONNX.js confirms a fall with temporal smoothing.
    """
    try:
        incident_data = {
            "type": data.get("type", "fall"),
            "confidence": data.get("confidence", 0.0),
            "inference_ms": data.get("inference_ms", 0),
            "source": data.get("source", "client_onnx"),
            "created_at": "now()"
        }
        supabase.table("incidents").insert(incident_data).execute()

        # Trigger email alert
        try:
            send_incident_email(data.get("type", "fall"), data.get("confidence", 0.0))
        except Exception as e:
            print(f"Email alert failed: {e}")

        return {"status": "logged"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "SafeWatch API is running",
        "models": {
            "yolo_pose": app.state.yolo_pose_model is not None,
            "fall_classifier": app.state.fall_model is not None,
            "scaler": app.state.scaler is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
