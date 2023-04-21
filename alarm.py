import os
import tkinter as tk
from tkinter import messagebox


def alarm():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Alarm", "坐正了")
    print("Alarm!")
    #os.system("alarm.mp3")  # play alarm sound
    return 0
alarm()