# deepfake_vit_efficient_roi

1. 전역으로 전체 이미지, 국소로 dlib을 통한 얼굴 좌표에 해당하는 각 부분의 패치 -> 64개  => 11/3
2. 이를 efficient net => ViT로 통과시킴
3. 이후에 이렇게 임베딩된 것을 프레임 단위로 열거하여 LSTM에 전달
4. LSTM 이후 연결된 dense 네트워크로 최종 분류  => 11/5

## 📊 논리적 구조
┌─────────────────────────────────────────┐
│ Input Image (224x224)                   │
└─────────────────────────────────────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
    Block 1  Block 1  Block 16
    (24ch)   (24ch)   (1280ch)
        │        │        │
        ▼        ▼        ▼
   Large     ROI      Small
   (4x4)  (97개 7x7) (32x32)
  Patch 56  Patch 7  Patch 7
        │        │        │
        ▼        ▼        ▼
   Dim 384  Dim 256  Dim 192