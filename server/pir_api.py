from fastapi import FastAPI
from logic.pir_trigger import PIRTrigger

app = FastAPI(title="PIR Sensor API")

pir = PIRTrigger(active_duration=8)

@app.post("/pir/trigger")
def trigger_pir():
    """
    เรียกเมื่อ PIR sensor ตรวจจับการเคลื่อนไหว
    """
    pir.trigger()
    return {
        "status": "TRIGGERED",
        "active_for": pir.remaining_time()
    }

@app.get("/pir/state")
def pir_state():
    """
    ใช้ให้ client (main.py) เช็คสถานะ
    """
    return {
        "active": pir.is_active(),
        "remaining": pir.remaining_time()
    }

@app.get("/")
def root():
    return {"status": "PIR API running"}
