from tkinter import messagebox
from Start import Start
import platform


def on_closing():
    if messagebox.askokcancel("Подтверждение закрытия", "Вы уверены, что хотите закрыть приложение?"):
        app.destroy()


app = Start()
app.geometry(f"{app.winfo_screenwidth()}x{app.winfo_screenheight()}")
app.protocol("WM_DELETE_WINDOW", on_closing)
app.mainloop()

system = platform.system()
if system == "Windows":
    app.after(0, lambda: app.state('zoomed'))
elif system == "Linux":
    app.attributes("-fullscreen", True)
app.bind("<F12>", lambda event: app.attributes("-fullscreen",
                                                         not app.attributes("-fullscreen")))
app.bind("<Escape>", lambda event: app.attributes("-fullscreen", False))
