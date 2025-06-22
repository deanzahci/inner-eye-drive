from pymongo import MongoClient
from datetime import datetime, timezone
import time

client = MongoClient("mongodb://localhost:27017/")
db = client["neuroguardian"]
collection = db["driver_insights"]

while True:
    data = {
        "timestamp": datetime.now(timezone.utc),
        "eeg_state": 2,        # Replace with real-time EEG reading
        "distraction": 1       # Replace with real-time distraction flag
    }
    collection.insert_one(data)
    print("Data inserted at", data["timestamp"])
    time.sleep(1)
