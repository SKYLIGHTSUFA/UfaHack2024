import pickle
import customtkinter as ctk
import os
import cv2
from PIL import Image
from deepface import DeepFace
from mtcnn import MTCNN
from catboost import CatBoostClassifier
import pandas as pd


class Predict(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.counter = 0
        self.storage_name: str
        self.grid_columnconfigure(0, weight=1)
        self.detector = MTCNN()
        self.catboost_model_usa = CatBoostClassifier()
        self.catboost_model_usa.load_model("../catboost_usa.cbm")
        self.catboost_model_ussr = CatBoostClassifier()
        self.catboost_model_ussr.load_model("../catboost_ussr.cbm")
        with open('../model/saved_dictionary.pkl', 'rb') as f:
            self.name_usa = pickle.load(f)
        with open('../model/saved_dictionary_russia.pkl', 'rb') as f:
            self.name_ussr = pickle.load(f)
        self.label = ctk.CTkLabel(self, text="Photo", fg_color="blue", text_color="white")
        self.label.grid(row=0, column=0, sticky="ew")
        self.button_get_dir = ctk.CTkButton(self, text="Choose folder", command=self.__open_file_dialog)
        self.button_get_dir.grid(row=1, column=0, pady=10, sticky="ew")
        self.button_start_predict = ctk.CTkButton(self, text="Start predict", command=self.__start_predict)
        self.button_start_predict.grid(row=2, column=0, pady=10, sticky="ew")

    def __open_file_dialog(self):
        root = ctk.CTk()
        root.withdraw()
        file_path = ctk.filedialog.askdirectory(title='Choose image dataset')
        if file_path != '':
            root.destroy()
            self.storage_name = file_path
            self.iterator = iter(os.listdir(self.storage_name))
            return file_path

    def __start_predict(self):
        if self.counter > 0:
            self.label.destroy()
        self.counter+=1
        filename = next(self.iterator)
        dicter3 = {}
        if filename.endswith("png") or filename.endswith("jpg"):
            print(filename)
            img = cv2.cvtColor(cv2.imread(os.path.join(self.storage_name, filename)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1080, 720))
            detections = self.detector.detect_faces(img)
            if len(detections) > 1:
                print(f'len detection = {len(detections)}')
            for detection in detections:
                confidence = detection["confidence"]#
                if confidence > 0.9:
                    x, y, w, h = detection["box"]
                    detected_face = img[int(y):int(y + h), int(x):int(x + w)]##
                    image = cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                    embedding = DeepFace.represent(detected_face, model_name='Facenet', enforce_detection=False)
                    ebd = embedding[0]["embedding"]
                    dicter3[1] = ebd
                    data_usa = pd.DataFrame.from_dict(dicter3.items())
                    data_usa.rename(columns={0: "id", 1: "embd"}, inplace=True, errors="ignore")
                    new_cols = pd.DataFrame(data_usa['embd'].apply(pd.Series))
                    df_usa = pd.concat([data_usa, new_cols], axis=1)
                    df_usa.drop(["embd"], axis=1, inplace=True, errors="ignore")
                    X = df_usa.drop(["id"], axis=1)
                    result = self.catboost_model_usa.predict(X)
                    result_ussr = self.catboost_model_ussr.predict(X)
                    index_usa = result[0][0]
                    index_ussr = result_ussr[0][0]
                    for images in os.listdir(f'C://Users//fatik//PycharmProjects//UfaHack2024//data//actors_usa//{index_usa}'):
                        image_usa = cv2.imread(os.path.join(f'C://Users//fatik//PycharmProjects//UfaHack2024//data//actors_usa//{index_usa}', images))
                        break
                    for images in os.listdir(f'C://Users//fatik//PycharmProjects//UfaHack2024//data//actors_ussr_russia//{index_ussr}'):
                        image_ussr = cv2.imread(os.path.join(f'C://Users//fatik//PycharmProjects//UfaHack2024//data//actors_ussr_russia//{index_ussr}', images))
                        break
                    image[0:128, 0:128] = cv2.resize(cv2.cvtColor(image_usa, cv2.COLOR_BGR2RGB), (128,128))
                    image[0:128, 138:266] = cv2.resize(cv2.cvtColor(image_ussr, cv2.COLOR_BGR2RGB), (128,128))
                    self.image = Image.fromarray(image)
                    self.image_tk = ctk.CTkImage(self.image, size=(self.image.width, self.image.height))
                    self.label = ctk.CTkLabel(self, image=self.image_tk, text=self.name_usa[result[0][0]], text_color="red")
                    self.label.grid(row=3, column=0, pady=10, sticky="ew")

