# korean-handwriting
한글 손글씨 생성 AI
![Preview](preview.gif)

# Tree
```
.
├── where-is-wally/
│   ├── data/
│   │   ├── fonts/              : 폰트
│   │   ├── imgs/               : 손글씨 폰트 numpy 배열
│   │   ├── gen_font_data.py    : 폰트(.ttf)를 numpy 배열로 변환
│   │   ├── generator.py        : 제너레이터
│   │   └── sampling.py         : 학습에 사용할 폰트 샘플링
│   ├── gan/
│   │   ├── gan.py              : GAN generator, discriminator
│   │   └── optimizer.py        : optimizer, loss
│   ├── models/                 : 모델
│   └── utils/
│       └── params.py           : 상수
└── train.py                    : 모델 훈련
```

# Skills
- python 3.7
- Numpy
- TensorFlow
- Pillow
- tqdm

# How to use  
### Set up
[폰트](https://clova.ai/handwriting/list.html)별 한글 이미지 생성  
```python 
$ python data/gen_font_data.py
```

### Training
```python 
$ python train.py
```
또는 아래 옵션 설정 가능
```
--imgs : 학습 데이터 경로
--model : 모델 저장 경로
--samples : 학습에 사용할 폰트 갯수
            (지정하지 않으면 모든 폰트 사용)
--batch-size : batch size
--epochs : epochs
```

### Generating
```
$ python gen_korean_handwriting.py
```
또는 아래 옵션 설정 가능
```
--model : 모델 경로
--output : 결과 저장 경로
--count : 생성할 손글씨 이미지 갯수
```
  
결과는 기본적으로 `data/results` 폴더 하위에  
현재 날짜와 시간으로 된 폴더를 생성하여 저장됨

<br>

---
  
<br>

#### Open Source License는 [이곳](NOTICE.md)에서 확인해주시고, 문의사항은 [Issue](https://github.com/IllIIIllll/korean-handwriting/issues) 페이지에 남겨주세요.
