import threading
from collections import deque

# Simple in-memory log buffer (thread-safe)
class LogBuffer:
    def __init__(self, maxlen=100):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
    def add(self, message):
        with self.lock:
            self.buffer.append(message)
    def get_all(self):
        with self.lock:
            return list(self.buffer)

log_buffer = LogBuffer()

def log_event(message):
    log_buffer.add(message)

def get_logs():
    return log_buffer.get_all()
