# 모델 정보 및 라이센스 가이드 

---
### 1. 목표
+ 경추 CT 영상 데이터를 활용하여 인공 디스크 치환술을 위한 bone segmentation 모델 개발
+ 수집된 CT 데이터와 경추체 영역에 대한 Polygon 라벨링을 학습 데이터로 사용하여, 경추체를 분할하기 위한 인공지능 기반의 이진 분할 모델
---

### 2. 라이브러리 요구 버전

|라이브러리|버전|
|:---:|:---:|
|Tensorflow|2.10.0|
|Opencv|4.6.0|
|Pydicom|2.4.2|
|SimpleITK|2.2.1|
|Tqdm|4.64.1|

---
### 3. Model description

#### 3.1. 모델
+ U-Net


#### 3.2. 입력데이터
+ 수집된 C-Spine CT 데이터 (.dcm)
+ C-spine 영역이 polygon 좌표로 라벨링된 데이터 (.json)

#### 3.3. 전처리
+ image 정방형으로 zero-padding
+ image resampling 256x256
+ CT image windowing (W:1800, L:400)
+ 딥러닝 학습을 위해 0.0~1.0 사잇값으로 normalize 

#### 3.4. 모델 구조
+ Input : 256x256 크기의 C-Spine CT image
+ Output : Segmented C-Spine binary image

![image](https://github.com/SaeByeolMun/git_test/assets/81259806/006cbd35-27ed-49f9-9c50-86746d0eff44)

#### 3.5. Training
+ 학습 데이터 세트

|    |Train|Validation|Test|Total|
|:---:|:---:|:---:|:---:|:---:|
|Slice|507,477|63,435|63,435|634,347|
|Ratio|80|10|10|100|

+ Hyper-parameter
  - Loss function : tversky_loss
  - Optimizer : Adam
  - Learning rate : 0.001
  - Batch size : 32

+ Evaluation
  - 성능 기준 지표는 이미지 등의 Segmentation에서 많이 쓰이는 지표인 DSC로, 계산식은 다음과 같음.
    
  - ![image](https://github.com/SaeByeolMun/git_test/assets/81259806/0e5f8cd5-9b3b-497f-bb3f-15d4a7219531)
  
  - 개발된 모델의 성능은, 92.67%로 우수한 성능을 나타냄. 

