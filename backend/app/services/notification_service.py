import resend
from app.core.config import settings
from datetime import datetime, timedelta

# Simple in-memory cooldown tracker (In production, use Redis or DB)
last_notification_time = {}

def send_incident_email(incident_type: str, confidence: float):
    resend.api_key = settings.RESEND_API_KEY
    
    # Cooldown check
    now = datetime.now()
    if incident_type in last_notification_time:
        if now < last_notification_time[incident_type] + timedelta(minutes=settings.NOTIFICATION_COOLDOWN):
            print(f"Notification for {incident_type} is on cooldown.")
            return False

    params = {
        "from": settings.EMAIL_FROM,
        "to": [settings.EMAIL_TO],
        "subject": f"⚠️ SafeWatch Alert: {incident_type.capitalize()} Detected!",
        "html": f"<strong>Incident Alert</strong><p>Type: {incident_type}</p><p>Confidence: {confidence:.2f}</p><p>Time: {now.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    }

    try:
        resend.Emails.send(params)
        last_notification_time[incident_type] = now
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
