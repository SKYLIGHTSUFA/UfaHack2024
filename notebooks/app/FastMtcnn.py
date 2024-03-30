import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time
from catboost import CatBoostClassifier
from tqdm.notebook import tqdm
from deepface import DeepFace
import pandas as pd
import pickle
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FastMTCNN(object):
    """Fast MTCNN implementation."""

    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.

        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.

        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        self.catboost_model_usa = CatBoostClassifier()
        self.catboost_model_usa.load_model("../catboost_usa.cbm")
        with open('../model/saved_dictionary.pkl', 'rb') as f:
            self.name_usa = pickle.load(f)

    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]

        boxes, probs = self.mtcnn.detect(frames[::self.stride])
        dicter3 = {}
        faces = []
        names = {}
        all_x = pd.DataFrame()
        for i, frame in enumerate(frames[::self.stride]):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
                image = frame[box[1]:box[3], box[0]:box[2]]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                embedding = DeepFace.represent(image, model_name='Facenet', enforce_detection=False)
                try:
                    ebd = embedding[0]["embedding"]
                except:
                    continue
                dicter3[1] = ebd
                data_usa = pd.DataFrame.from_dict(dicter3.items())
                data_usa.rename(columns={0: "id", 1: "embd"}, inplace=True, errors="ignore")
                new_cols = pd.DataFrame(data_usa['embd'].apply(pd.Series))
                df_usa = pd.concat([data_usa, new_cols], axis=1)
                df_usa.drop(["embd"], axis=1, inplace=True, errors="ignore")
                df_usa["id"] = df_usa["id"].apply(lambda x: str(x)[:str(x).find("_")])
                y, X = df_usa["id"], df_usa.drop(["id"], axis=1)
                all_x = pd.concat([all_x, X], axis=0)
        result = self.catboost_model_usa.predict(all_x)
        vals, counts = np.unique(result, return_counts=True)
        return vals[np.argmax(counts)]