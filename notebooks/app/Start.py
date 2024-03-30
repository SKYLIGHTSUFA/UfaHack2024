import customtkinter
from Predict_photo import Predict
from Predict_video import PredictV


class Start(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        self.title("Face recognition system")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.predict = Predict(self)
        self.predict.grid(row=0, column=0, sticky="nswe", pady=5)
        self.video= PredictV(self)
        self.video.grid(row=0, column=1, sticky="nsew", pady=5)
