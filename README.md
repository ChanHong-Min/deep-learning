# 프로젝트 개요

이 프로젝트는 여러 개의 Python 스크립트를 포함하며, 각각의 파일은 다양한 머신 러닝 및 딥 러닝 개념을 다루고 있다. 각 파일의 역할 및 구현된 내용은 아래와 같다.

---

## 📂 파일 설명

### 1️⃣ `hw2.py`
- **내용:** 소프트맥스 분류기를 사용한 다중 클래스 분류 문제 해결.
- **주요 개념:** 
  - 크로스 엔트로피 손실 함수
  - 선형 분류기 구현 및 성능 평가
  - `scipy.optimize.minimize`를 이용한 최적화

---

### 2️⃣ `hw3.py`
- **내용:** 기본적인 신경망 계층(layer) 구현.
- **주요 개념:** 
  - 선형 계층 (`nn_linear_layer`)
  - 활성화 함수 계층 (`nn_activation_layer`)
  - 소프트맥스 계층 (`nn_softmax_layer`)
  - 크로스 엔트로피 손실 함수 (`nn_cross_entropy_layer`)
  - 역전파(Backpropagation) 및 가중치 업데이트

---

### 3️⃣ `hw4.py`
- **내용:** 합성곱 신경망(CNN)의 주요 구성 요소 구현.
- **주요 개념:** 
  - `view_as_windows`: 슬라이딩 윈도우를 사용한 입력 텐서 변환
  - 합성곱 계층 (`nn_convolutional_layer`)
  - 맥스 풀링 계층 (`nn_max_pooling_layer`)

---

### 4️⃣ `hw5.py`
- **내용:** MNIST 데이터셋을 사용한 이미지 분류 모델.
- **주요 개념:** 
  - `keras.datasets.mnist`에서 데이터 로드
  - 기본 CNN 모델 (`nn_mnist_classifier`) 및 PyTorch 기반 모델 (`MNISTClassifier_PT`)
  - SGD(확률적 경사 하강법)과 크로스 엔트로피 손실을 사용한 학습

---

### 5️⃣ `hw7.py`
- **내용:** Transformer 기반 감성 분석 모델.
- **주요 개념:** 
  - 다중 헤드 어텐션 (`MultiHeadAttention`)
  - 트랜스포머 인코더 블록 (`TF_Encoder_Block`)
  - 포지셔널 인코딩 (`PosEncoding`)
  - `sentiment_classifier`: IMDB 데이터셋을 활용한 감성 분석 모델
