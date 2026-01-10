import requests

class PIRClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def is_active(self) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}/pir/state",
                timeout=0.2
            )
            return r.json().get("active", False)
        except Exception as e:
            print("PIR API error:", e)
            return False
