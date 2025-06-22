import tkinter as tk
from tkinter import ttk
import webbrowser
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import asyncio
import websockets
import json
import time
from collections import deque

import client

is_started = False
client_thread = None
eeg_thread = None

# EEG data storage
eeg_data = deque(maxlen=100)  # Store last 100 EEG readings
time_data = deque(maxlen=100)
current_eeg_state = 0

class EEGClient:
    def __init__(self, update_callback):
        self.update_callback = update_callback
        self.running = False
        self.loop = None
    
    async def connect_to_eeg_service(self):
        uri = "ws://127.0.0.1:8002/ws"
        try:
            async with websockets.connect(uri, ping_timeout=20, ping_interval=20) as websocket:
                print("[EEG] Connected to EEG service")
                await websocket.send("hello")
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        if 'state' in data:
                            self.update_callback(data['state'])
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("[EEG] Connection closed")
                        break
                    except json.JSONDecodeError:
                        print(f"[EEG] Invalid JSON: {message}")
        except Exception as e:
            print(f"[EEG] Connection failed: {e}")
    
    def start(self):
        self.running = True
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.connect_to_eeg_service())
        except Exception as e:
            print(f"[EEG] Error: {e}")
        finally:
            self.loop.close()
    
    def stop(self):
        self.running = False
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)

eeg_client = None

def update_eeg_data(state_value):
    """Callback function to update EEG data from WebSocket"""
    global eeg_data, time_data, current_eeg_state
    
    current_time = time.time()
    current_eeg_state = state_value
    
    eeg_data.append(state_value)
    time_data.append(current_time)
    
    # Update the plot (this will be called from a different thread)
    root.after(0, update_plot)

def update_plot():
    """Update the matplotlib plot with new EEG data"""
    if len(eeg_data) > 0:
        ax.clear()
        
        # Convert time data to relative seconds for better visualization
        if len(time_data) > 0:
            start_time = time_data[0]
            relative_times = [(t - start_time) for t in time_data]
        else:
            relative_times = list(range(len(eeg_data)))
        
        # Plot EEG data
        ax.plot(relative_times, list(eeg_data), 'b-', linewidth=2, marker='o', markersize=3)
        
        ax.set_title('Real-time EEG State Monitoring', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('EEG State')
        ax.set_ylim(-0.5, 3.5)
        ax.grid(True, alpha=0.3)
        
        # Add colored zones for different states
        ax.axhspan(-0.5, 0.5, alpha=0.1, color='green', label='Alert')
        ax.axhspan(0.5, 1.5, alpha=0.1, color='yellow', label='Normal')
        ax.axhspan(1.5, 2.5, alpha=0.1, color='orange', label='Tired')
        ax.axhspan(2.5, 3.5, alpha=0.1, color='red', label='Drowsy')
        
        # Add current state indicator
        if len(eeg_data) > 0:
            latest_value = eeg_data[-1]
            ax.axhline(y=latest_value, color='black', linestyle='--', alpha=0.8, linewidth=1)
            ax.text(0.02, 0.98, f'Current State: {latest_value:.1f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        canvas.draw()

def get_state_description(state):
    """Get human-readable description of EEG state"""
    if state <= 0.5:
        return "Alert & Focused"
    elif state <= 1.5:
        return "Normal Attention"
    elif state <= 2.5:
        return "Getting Tired"
    else:
        return "Drowsy - Alert Needed!"

def update_button():
    if is_started:
        start_stop_button.config(text="Stop", command=stop)
    else:
        start_stop_button.config(text="Start", command=start)

def start():
    global is_started, client_thread, eeg_thread, eeg_client
    print("Starting services...")
    is_started = True
    
    # Run main client in a separate thread
    client_thread = threading.Thread(target=client.start, daemon=True)
    client_thread.start()
    
    # Run EEG client in a separate thread for the plot
    eeg_client = EEGClient(update_eeg_data)
    eeg_thread = threading.Thread(target=eeg_client.start, daemon=True)
    eeg_thread.start()
    
    update_button()

def stop():
    global is_started, client_thread, eeg_client, eeg_thread
    print("Stopping services...")
    is_started = False
    
    # Stop main client
    client.stop()
    if client_thread and client_thread.is_alive():
        client_thread.join(timeout=2)
    
    # Stop EEG client
    if eeg_client:
        eeg_client.stop()
    if eeg_thread and eeg_thread.is_alive():
        eeg_thread.join(timeout=2)
    
    update_button()

root = tk.Tk()
root.title("Inner Eye Drive - EEG Monitor")
root.geometry("900x700")

# Configure grid weights so the plot can expand
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(1, weight=1)

# Create control frame
control_frame = ttk.Frame(root)
control_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

# Create the start/stop button
start_stop_button = ttk.Button(control_frame, text="Start", command=start)
start_stop_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

# Create a status label
status_label = ttk.Label(control_frame, text="Status: Stopped")
status_label.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

# Configure control frame grid weights
control_frame.grid_columnconfigure(0, weight=1)
control_frame.grid_columnconfigure(1, weight=1)

# Create matplotlib figure and canvas for EEG data
fig = Figure(figsize=(12, 6), dpi=100)
ax = fig.add_subplot(111)
ax.set_title('Real-time EEG State Monitoring', fontsize=14, fontweight='bold')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('EEG State')
ax.set_ylim(-0.5, 3.5)
ax.grid(True, alpha=0.3)

# Add initial state zones
ax.axhspan(-0.5, 0.5, alpha=0.1, color='green', label='Alert')
ax.axhspan(0.5, 1.5, alpha=0.1, color='yellow', label='Normal')
ax.axhspan(1.5, 2.5, alpha=0.1, color='orange', label='Tired')
ax.axhspan(2.5, 3.5, alpha=0.1, color='red', label='Drowsy')

# Create canvas and add to tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Update status label based on EEG state
def update_status():
    if is_started:
        state_desc = get_state_description(current_eeg_state)
        status_label.config(text=f"Status: {state_desc}")
        
        # Change color based on EEG state
        if current_eeg_state <= 1:
            status_label.config(foreground="green")
        elif current_eeg_state <= 2:
            status_label.config(foreground="orange") 
        else:
            status_label.config(foreground="red")
    else:
        status_label.config(text="Status: Stopped", foreground="black")
    
    # Schedule next update
    root.after(500, update_status)

# Start status updates
update_status()

if __name__ == "__main__":
    root.mainloop()