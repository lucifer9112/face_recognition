# config.py

FACE_XML_PATH = "haarcascade_frontalface_default.xml"

DATA_DIR = "face_database"
MODEL_PATH = DATA_DIR + "/lbph_model.yml"
LABELS_PATH = DATA_DIR + "/labels.pkl"
LOGS_PATH = DATA_DIR + "/recognition_logs.json"

MIN_FACE_SIZE = (50, 50)
CONFIDENCE_THRESHOLD = 35.0  # LBPH distance threshold (lower = stricter). 35-45 is typical

# CaffeNet models paths
CAFFE_PROTOTXT_8 = "ultimate_caffenet_deploy.prototxt"
CAFFE_MODEL_8 = "ultimate_caffenet.caffemodel"
CAFFE_PROTOTXT_BIN = "ultimate_caffenet_binary_deploy.prototxt"
CAFFE_MODEL_BIN = "ultimate_caffenet.caffemodel"
