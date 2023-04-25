import os
import time
import tkinter as tk
from tkinter import messagebox


def alarm():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo("Alarm", "坐正了")
    #root.mainloop()
    print("Alarm!")
    #os.system("alarm.mp3")  # play alarm sound
    return 0
alarm()