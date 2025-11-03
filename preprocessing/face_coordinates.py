import cv2
# 최신 버전에서는 이렇게 불러옵니다
from mediapipe.python.solutions import face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
import os
import matplotlib.pyplot as plt

# 이미지 불러오기
DFD_PATH = "../data/archive/FaceForensics++_C23/crops/DeepFakeDetection"
image_folders = os.listdir(DFD_PATH)
sample_folder = image_folders[0]
sample_folder_path = os.path.join(DFD_PATH, sample_folder)
sample_file = os.listdir(sample_folder_path)[0]
image_path = os.path.join(sample_folder_path, sample_file)
print(image_path)
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# FaceMesh 객체 생성
with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as mesh:

    results = mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            print("총 랜드마크 개수:", len(face_landmarks.landmark[::4]))
            for idx, lm in enumerate(face_landmarks.landmark[::4]):
                x = int(lm.x * image.shape[1])
                y = int(lm.y * image.shape[0])
                print(f"랜드마크 {idx}: x={x}, y={y}")

                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

final_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(final_image_rgb)
plt.show()