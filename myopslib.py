# built-in
import os
import logging

import requests
import numpy as np
import pandas as pd
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

def load_model(p="./mlp-model"):
    """ 모델을 로드하는 함수 """
    model = tf.keras.models.load_model(p)
    return model

def load_dataset(p='./auto-mpg.csv'):
    """ 데이터를 로드하는 함수 """
    if not os.path.exists(p):
        logging.info("데이터 셋이 존재하지 않습니다. 다운로드를 시작합니다.")
        url = "https://raw.githubusercontent.com/ProtossDragoon/flask-docker/master/notebooks/auto-mpg.csv"
        r = requests.get(url, allow_redirects=True, timeout=5)
        open(p, "wb").write(r.content)
        logging.info("데이터 셋 다운로드 완료: %s", p)
    dataset = pd.read_csv(p)
    return dataset

def retrain(p_new="./mlp-model"):
    """ 모델을 추가 학습하는 함수"""
    # 데이터 로드
    dataset = load_dataset()

    # 모델 로드
    model = load_model()

    # 훈련 data set과 평가 data set 분리
    train_dataset = dataset.sample(frac=0.8, random_state=0) 
    test_dataset = dataset.drop(train_dataset.index)

    # feature와 label 분리
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    # 추가 학습
    model.fit(train_features, train_labels, validation_split=0.2, epochs=5)

    # 모델 평가
    test_result = model.evaluate(test_features, test_labels, verbose=0)
    logging.debug("추가 학습 MAE: %s", test_result)

    # 모델 저장
    model.save(p_new)

    return test_result

def get_required_features() -> set:
    """ 필요한 특징명 집합을 반환하는 함수 """
    return {
        "Cylinders",
        "Displacement",
        "Horsepower",
        "Weight",
        "Acceleration",
        "ModelYear",
        "Country"
    }

def format_x(payload) -> np.ndarray:
    """ 입력받은 특징 값들을 넘파이 배열로 변환하는 함수: inc. onehot-encoding for country """
    country = {
        "Europe": 0.0,
        "Japan": 0.0,
        "USA": 0.0
    }
    if payload['Country'] == "Europe":
        country['Europe'] = 1
    if payload['Country'] == "Japan":
        country['Japan'] = 1
    if payload['Country'] == "USA":
        country['USA'] = 1

    return np.array(
        [
            float(payload["Cylinders"]),
            float(payload["Displacement"]),
            float(payload["Horsepower"]),
            float(payload["Weight"]),
            float(payload["Acceleration"]),
            float(payload["ModelYear"]),
            float(country["Europe"]),
            float(country["Japan"]),
            float(country["USA"])
        ]
    )

def format_y(y_tensor) -> list:
    """ 추론 결과를  python 실수가 담긴 리스트로 변환하는 함수 """
    y = list(y_tensor[0].numpy())
    y = [ float(item) for item in y]

    return y

def predict(x_json) -> list:
    """ 입력 받은 특징 값들을 사용해 자동차의 연비를 추론하는 함수
    
    Input sample:
    {
        "Cylinders": 8,
        "Displacement": 390.0,
        "Horsepower": 190,
        "Weight": 3850,
        "Acceleration": 8.5,
        "ModelYear": 70,
        "Country": "USA",
    }

    Output sample:
    [ 16.075947 ]
    """
    # laod model
    model = load_model()

    # 함수 입력 값을 numpy array로 변환
    x = format_x(x_json)

    # 모델 추론
    y_tensor = model(x)

    # 모델 추론 결과 후처리
    y = format_y(y_tensor)

    return y