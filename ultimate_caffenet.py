# ultimate_caffenet.py

import cv2
import numpy as np
from config import (
    CAFFE_PROTOTXT_8,
    CAFFE_MODEL_8,
    CAFFE_PROTOTXT_BIN,
    CAFFE_MODEL_BIN,
)


class UltimateCaffeNet8:
    """8-class classifier (UltimateCaffeNet)"""

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTXT_8, CAFFE_MODEL_8)
        # Apne 8 class labels yahan set karo:
        self.classes = [
            "class_0",
            "class_1",
            "class_2",
            "class_3",
            "class_4",
            "class_5",
            "class_6",
            "class_7",
        ]

    def _preprocess(self, face_bgr):
        # AlexNet-style: 227x227, BGR, mean subtraction
        blob = cv2.dnn.blobFromImage(
            face_bgr,
            scalefactor=1.0,
            size=(227, 227),
            mean=(104, 117, 123),
            swapRB=False,
            crop=False,
        )
        return blob

    def predict(self, face_bgr):
        """
        Returns: (label_str, confidence_float_0to1)
        """
        self.net.setInput(self._preprocess(face_bgr))
        probs = self.net.forward().flatten()  # prob layer (softmax) [web:91][web:95][web:101]
        cid = int(np.argmax(probs))
        conf = float(probs[cid])
        label = (
            self.classes[cid] if 0 <= cid < len(self.classes) else f"class_{cid}"
        )
        return label, conf


class UltimateCaffeNetBinary:
    """2-class (binary) classifier (UltimateCaffeNet-Binary)"""

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTXT_BIN, CAFFE_MODEL_BIN)
        # Binary labels: customize karo (e.g. ["fake", "real"] ya ["neg", "pos"])
        self.classes = ["neg", "pos"]

    def _preprocess(self, face_bgr):
        blob = cv2.dnn.blobFromImage(
            face_bgr,
            scalefactor=1.0,
            size=(227, 227),
            mean=(104, 117, 123),
            swapRB=False,
            crop=False,
        )
        return blob

    def predict(self, face_bgr):
        """
        Returns: (label_str, confidence_float_0to1, positive_prob_float_0to1)
        """
        self.net.setInput(self._preprocess(face_bgr))
        probs = self.net.forward().flatten()  # shape (2,) [web:91][web:98]
        cid = int(np.argmax(probs))
        conf = float(probs[cid])
        label = (
            self.classes[cid] if 0 <= cid < len(self.classes) else f"class_{cid}"
        )
        pos_prob = float(probs[1]) if len(probs) > 1 else conf
        return label, conf, pos_prob
