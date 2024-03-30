import pickle
import customtkinter as ctk
import os
from mtcnn import MTCNN
from catboost import CatBoostClassifier
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import time
from FastMtcnn import FastMTCNN
import threading

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PredictV(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.counter = 0
        self.storage_name: str
        self.grid_columnconfigure(0, weight=1)
        self.detector = MTCNN()
        self.catboost_model_usa = CatBoostClassifier()
        self.catboost_model_usa.load_model("../catboost_usa.cbm")
        self.fast_mtcnn = FastMTCNN(
            stride=32,
            resize=1,
            margin=14,
            factor=0.6,
            keep_all=True,
            device=device
        )
        self.catboost_model_usa.load_model("../catboost_usa.cbm")
        with open('../model/saved_dictionary.pkl', 'rb') as f:
            self.name_usa = pickle.load(f)
        self.label = ctk.CTkLabel(self, text="Video", fg_color="blue", text_color="white")
        self.label.grid(row=0, column=0, sticky="ew")
        self.button_start_predict = ctk.CTkButton(self, text="Start predict", command=self.run_detection)
        self.button_start_predict.grid(row=1, column=0, pady=10, sticky="ew")

    def __open_file_dialog(self):
        root = ctk.CTk()
        root.withdraw()
        file_path = ctk.filedialog.askdirectory(title='Choose image dataset')
        if file_path != '':
            root.destroy()
            self.storage_name = file_path
            self.iterator = iter(os.listdir(self.storage_name))
            return file_path

    def run_detection(self):
        frames = []
        batch_size = 256
        cap = cv2.VideoCapture(0)
        self.image_usa = cv2.resize(cap.read()[1], (1080, 720))
        while True:
            frame = cv2.resize(cap.read()[1], (1080, 720))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            im_h = cv2.hconcat([frame, self.image_usa])
            cv2.imshow("tester", im_h)
            if cv2.waitKey(1) == 27:
                break

            if len(frames) >= batch_size:
                index_usa = self.fast_mtcnn(frames)
                frames = []
                def set_image_all(index_usa):
                    print(index_usa)
                    for images in os.listdir(
                            f'C://Users//fatik//PycharmProjects//UfaHack2024//data//actors_usa//{index_usa}'):
                        self.image_usa = cv2.imread(
                            os.path.join(f'C://Users//fatik//PycharmProjects//UfaHack2024//data//actors_usa//{index_usa}', images))
                        self.image_usa = cv2.resize(self.image_usa, (1080, 720))
                        break
                    def set_image():
                        self.image = Image.fromarray(self.image_usa)
                        self.image_tk = ctk.CTkImage(self.image, size=(self.image.width, self.image.height))
                        self.label_image = ctk.CTkLabel(self, image=self.image_tk, text=self.name_usa[index_usa], text_color="red")
                        self.label_image.grid(row=2, column=0, pady=10, sticky="ew")
                    thread = threading.Thread(target=set_image)
                    thread.start()

                target1 = threading.Thread(target=set_image_all, args=(index_usa,))
                target1.start()

