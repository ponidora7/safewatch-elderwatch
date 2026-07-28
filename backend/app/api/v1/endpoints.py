from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.services.ai_service import query_hf_inference
from app.services.notification_service import send_incident_email
from app.core.config import settings
from supabase import create_client, Client

router = APIRouter()
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

class FrameInput(BaseModel):
    image: str  # Base64 encoded image

@router.post("/inference/frame")
async def process_frame(frame: FrameInput):
    # 1. Forward frame to AI Inference
    ai_result = query_hf_inference(frame.image)
    
    if ai_result.get("detected") and ai_result.get("confidence", 0) >= settings.CONFIDENCE_THRESHOLD:
        # 2. Store in Supabase
        incident_data = {
            "type": ai_result.get("type"),
            "confidence": ai_result.get("confidence"),
            "inference_ms": ai_result.get("inference_ms"),
            "created_at": "now()"
        }
        try:
            supabase.table("incidents").insert(incident_data).execute()
            print("Success: Incident logged to Supabase.")
        except Exception as e:
            print(f"Supabase logging failed: {e}")
            
        # 3. Send Email Notification (with cooldown)
        try:
            send_incident_email(ai_result.get("type"), ai_result.get("confidence"))
            print("Success: Email notification triggered.")
        except Exception as e:
            print(f"Email notification failed: {e}")

    return {
        "status": "processed",
        "ai_result": ai_result
    }
