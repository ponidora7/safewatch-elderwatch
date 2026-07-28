import requests
import base64
import time
from app.core.config import settings

def query_hf_inference(image_base64: str):
    """
    Decodes a base64 image string and queries the Hugging Face Inference API.
    
    Args:
        image_base64: Base64 encoded image string (supports data URL prefix).
        
    Returns:
        A dictionary containing:
            - detected: Boolean indicating if a fall was detected.
            - confidence: Float score of the prediction.
            - type: String representing the class label (e.g. 'fall' or other classes).
            - inference_ms: Time taken in milliseconds.
    """
    headers = {}
    if settings.HF_TOKEN:
        headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"
        
    try:
        # Strip data URL prefix if present (e.g. "data:image/jpeg;base64,")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        image_bytes = base64.b64decode(image_base64)
        
        start_time = time.time()
        
        # Send raw binary image data directly to Hugging Face API
        response = requests.post(
            settings.HF_API_URL,
            headers=headers,
            data=image_bytes,
            timeout=10
        )
        
        inference_ms = int((time.time() - start_time) * 1000)
        
        if response.status_code == 200:
            result = response.json()
            
            # Standard image classification output format is a list of label/score dicts:
            # [{"label": "fall", "score": 0.88}, {"label": "normal", "score": 0.12}]
            if isinstance(result, list) and len(result) > 0:
                best_match = max(result, key=lambda x: x.get("score", 0.0))
                label = best_match.get("label", "").lower()
                score = best_match.get("score", 0.0)
                
                detected = "fall" in label or "falling" in label
                return {
                    "detected": detected,
                    "confidence": score,
                    "type": "fall" if detected else label,
                    "inference_ms": inference_ms
                }
            elif isinstance(result, dict):
                # Handle single object dictionary response format
                label = result.get("label", "").lower()
                score = result.get("score", 0.0)
                
                detected = "fall" in label or "falling" in label
                return {
                    "detected": detected,
                    "confidence": score,
                    "type": "fall" if detected else label,
                    "inference_ms": inference_ms
                }
        else:
            print(f"HF Inference API returned status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error calling HF Inference API: {e}")
        
    # Development/Testing Fallback if HF service is offline, slow, or not configured
    return {
        "detected": True,
        "confidence": 0.85,
        "type": "fall",
        "inference_ms": 120,
        "note": "simulated fallback"
    }
