import time

class PIRTrigger:
    def __init__(self, active_duration=8):
        self.active_duration = active_duration
        self.active_until = 0.0

    def trigger(self):
        """เรียกเมื่อ PIR sensor ทำงาน"""
        self.active_until = time.time() + self.active_duration

    def is_active(self) -> bool:
        """สถานะปัจจุบันของ PIR"""
        return time.time() < self.active_until

    def remaining_time(self) -> float:
        return max(0.0, self.active_until - time.time())
    