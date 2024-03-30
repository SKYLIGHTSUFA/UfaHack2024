import pickle
import customtkinter as ctk
import os
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace
from mtcnn import MTCNN
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

class Server(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.counter = 0
        self.storage_name: str
        self.grid_columnconfigure(0, weight=1)
        self.detector = MTCNN()
        import socket
        self.s = socket.socket()
        host = "192.168.120.240"
        port = 12345
        self.s.bind((host, port))
        self.s.listen(5)
        self.catboost_model_usa = CatBoostClassifier()
        self.catboost_model_usa.load_model("../catboost_usa.cbm")
        with open('../model/saved_dictionary.pkl', 'rb') as f:
            self.name_usa = pickle.load(f)
       # self.grid_rowconfigure(0, weight=1)
        self.label = ctk.CTkLabel(self, text="Server", fg_color="blue", text_color="white")
        self.label.grid(row=0, column=0, sticky="ew")
        self.button_start_predict = ctk.CTkButton(self, text="Get photo from server", command=self.__start_predict)
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
    def __start_predict(self):
        while True:
            c = 0
            con, addr = self.s.accept()
            with open('FDJ.jpg', 'wb') as f:
                while True:
                    c+=1
                    print(1)
                    data = con.recv(65536)
                    if c >= 3:
                        f.write(data)
                        break
                    if not data:
                        f.write(data)
                        break
                        break
            #break

                    f.write(data)

                break

            #break

            print("break")
            break
        if self.counter > 0:
            self.label.destroy()
        self.counter+=1
        filename = 'FDJ.jpg'
        dicter3 = {}
        if filename.endswith("png") or filename.endswith("jpg"):
            print(filename)
            img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1080, 720))
            detections = self.detector.detect_faces(img)
            if len(detections) > 1:
                print("AAAAAAAAA")
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
                    #df_usa["id"] = df_usa["id"].apply(lambda x: str(x)[:str(x).find("_")])
                    y, X = df_usa["id"], df_usa.drop(["id"], axis=1)
                    result = self.catboost_model_usa.predict(X)
                    index_usa = result[0][0]
                    print(result)
                    string = self.name_usa[result[0][0]]
                    encoded_string = string.encode('utf-8')  # Кодируем строку в UTF-8
                    binary_representation = ''.join(format(byte, '08b') for byte in encoded_string)
                    con.send(binary_representation.encode('utf-8'))
                    con.close()
                    print(self.name_usa[result[0][0]])
                    for images in os.listdir(f'C://Users//fatik//PycharmProjects//UfaHack2024//data//actors_usa//{index_usa}'):
                        image_usa = cv2.imread(os.path.join(f'C://Users//fatik//PycharmProjects//UfaHack2024//data//actors_usa//{index_usa}', images))
                        break
                    image[0:128, 0:128] = cv2.resize(cv2.cvtColor(image_usa, cv2.COLOR_BGR2RGB), (128,128))
                    #print(np.argmax(result[0], axis=0))
                    #print(len(result[0]))
                    self.image = Image.fromarray(image)
                    self.image_tk = ctk.CTkImage(self.image, size=(self.image.width, self.image.height))
                    self.label = ctk.CTkLabel(self, image=self.image_tk, text=self.name_usa[result[0][0]], text_color="red")
                    self.label.grid(row=3, column=0, pady=10, sticky="ew")
