# 모델 정보 및 라이센스 가이드 

### 1. 목표
+ 경추 CT 영상 데이터를 활용하여 인공 디스크 치환술을 위한 bone segmentation 모델 개발
+ 수집된 CT 데이터와 경추체 영역에 대한 Polygon 라벨링을 학습 데이터로 사용하여, 경추체를 분할하기 위한 인공지능 기반의 이진 분할 모델

### 2. 라이브러리 요구 버전
|라이브러리|버전|
|:---:|:---:|
|Tensorflow|2.10.0|
|Opencv|4.6.0|
|Pydicom|2.4.2|
|SimpleITK|2.2.1|
|Tqdm|4.64.1|

### 3. Model description
#### 3.1. 모델
+ U-Net


#### 3.2. 입력데이터
+ 수집된 C-Spine CT 데이터 (.dcm)
+ C-spine 영역이 polygon 좌표로 라벨링된 데이터 (.json)
