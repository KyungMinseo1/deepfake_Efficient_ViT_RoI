import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
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

if __name__=='__main__':
  print(ROOT_DIR)