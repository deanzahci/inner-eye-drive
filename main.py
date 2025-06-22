import tkinter as tk
from tkinter import ttk
import webbrowser
import threading
import matplotlib.pyplot as plt

is_started = False

def start():
    global is_started
    print("Start button clicked!")
    is_started = True

def stop():
    global is_started
    print("Stop button clicked!")
    is_started = False

def open_github():
    webbrowser.open("https://youtube.com/")

root = tk.Tk()
root.title("My Tkinter App")
root.geometry("400x300")

# Configure grid weights so columns can expand
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

if is_started:
    button = ttk.Button(root, text="Stop", command=stop)
    button.grid(row=0, column=0, padx=10, pady=20, sticky="ew")
else:
    button = ttk.Button(root, text="Start", command=start)
    button.grid(row=0, column=0, padx=10, pady=20, sticky="ew")

button = tk.Button(root, text="Open GitHub", command=open_github)
button.grid(row=0, column=1, padx=10, pady=20, sticky="ew")

if __name__ == "__main__":
    root.mainloop()