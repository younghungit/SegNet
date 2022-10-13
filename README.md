# SegNet


## 진행 순서

0. 데이터 수집 
    - 저작권 없는 만화 데이터 (3000장)
1. 모델 생성 (SegNet 기반)
    - 레이어 변경 및 추가
2. 데이터 전처리
    - 흑백 전환
    - binary 형태로 이미지 전환
    - 사이즈 조정
2. 모델 학습 (Train)
    - learning rate 변경
    - Data Augmentation
    - optimizer 변경
    - scheduler 활용
3. 결과 후처리 
    - ocr 작업 (Pytesseract)
    - google translator API


## 봐야할 파일

    - convert.py
    - dataset.py
    - model.py
    - model2.py
    - postprocess.py