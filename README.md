# deepfake_vit_efficient_roi
멀티스케일 패치 추출과 ROI 기반 임베딩을 결합한 DeepFake 탐지 모델입니다.
EfficientNet과 Vision Transformer(ViT)를 결합하고, 시계열 프레임 단위 특징을 LSTM으로 처리하여 최종 분류를 수행합니다.

1. 전역으로 전체 이미지, 국소로 dlib을 통한 얼굴 좌표에 해당하는 각 부분의 패치 -> 64개  => 11/3
2. 이를 efficient net => ViT로 통과시킴
3. 이후에 이렇게 임베딩된 것을 프레임 단위로 열거하여 LSTM에 전달
4. LSTM 이후 연결된 dense 네트워크로 최종 분류  => 11/5

## 📊 논리적 구조

┌─────────────────────────────────────────┐
│             Input Image (224×224)       │
└─────────────────────────────────────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
    Block 1   Block 1   Block 16
    (24ch)    (24ch)    (1280ch)
        │         │         │
        ▼         ▼         ▼
   Large       ROI         Small
   (4×4)    (97개 7×7)   (32×32)
   Patch 56   Patch 7     Patch 7
        │         │         │
        ▼         ▼         ▼
   Dim 384     Dim 256     Dim 192

## 데이터 흐름

Input (224×224)
   │
   ├──▶ EfficientNet Backbone
   │         │
   │         ├──▶ ViT Embedding (Patch-level)
   │         │
   └──▶ Multi-Scale Feature Fusion
             │
             ▼
        LSTM (Temporal)
             │
             ▼
         Dense Layers
             │
             ▼
         DeepFake / Real
