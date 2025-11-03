import os

ROOT_DIR = os.path.join(os.path.abspath(__file__), '..', '..')

DATA_DIR = os.path.join(ROOT_DIR, 'data')
FACE_FORENSICS = os.path.join(DATA_DIR, 'archive', 'FaceForensics++_C23')
CSV =  os.path.join(FACE_FORENSICS, 'csv')
DFD = os.path.join(FACE_FORENSICS, 'DeepFakeDetection')
DFS = os.path.join(FACE_FORENSICS, 'Deepfakes')
F2F = os.path.join(FACE_FORENSICS, 'Face2Face')
FSH = os.path.join(FACE_FORENSICS, 'FaceShifter')
FSW = os.path.join(FACE_FORENSICS, 'FaceSwap')
NTX = os.path.join(FACE_FORENSICS, 'NeuralTextures')
ORG = os.path.join(FACE_FORENSICS, 'original')