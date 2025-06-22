import tkinter as tk
from tkinter import ttk
import webbrowser
import threading
import matplotlib.pyplot as plt

import client

is_started = False
client_thread = None

def update_button():
    if is_started:
        start_stop_button.config(text="Stop", command=stop)
    else:
        start_stop_button.config(text="Start", command=start)

def start():
    global is_started, client_thread
    print("Starting client...")
    is_started = True
    # Run client in a separate thread to avoid blocking the GUI
    client_thread = threading.Thread(target=client.start, daemon=True)
    client_thread.start()
    update_button()

def stop():
    global is_started, client_thread
    print("Stopping client...")
    is_started = False
    client.stop()
    if client_thread and client_thread.is_alive():
        # Give the thread a moment to clean up
        client_thread.join(timeout=2)
    update_button()

root = tk.Tk()
root.title("My Tkinter App")
root.geometry("400x300")

# Configure grid weights so columns can expand
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Create the start/stop button
start_stop_button = ttk.Button(root, text="Start", command=start)
start_stop_button.grid(row=0, column=0, padx=10, pady=20, sticky="ew")

if __name__ == "__main__":
    root.mainloop()