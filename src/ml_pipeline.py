import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import shap

def import_training_data(data_path, random_state):
    training_data = pd.read_csv(
        data_path / 'training_data.csv',
        usecols=[
            'category_id',
            'price',
            'is_first_time_customer',
            'order_day_of_week',
            'price_percentile',
            'returned'
        ]
    )
    X = training_data[[
        'category_id',
        'price',
        'is_first_time_customer',
        'order_day_of_week',
        'price_percentile'
    ]]
    Y = training_data['returned']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train, random_state, data_path):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        n_estimators=100,
        random_state=random_state,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=9,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    model.save_model(data_path / 'xgb_model.ubj')

    return model

def test_model(model, decision_threshold, x_test, y_test):
    predictions = model.predict_proba(x_test)
    predictions = np.where(predictions[:, 1] > decision_threshold, 1, 0)

    pred_confusion_matrix = confusion_matrix(y_test, predictions)

    return classification_report(y_test, predictions), pred_confusion_matrix


if __name__ == '__main__':
    data_path = Path(__file__).resolve().parent.parent / 'data'
    random_state = 42
    decision_threshold = 0.47

    x_train, x_test, y_train, y_test = import_training_data(data_path, random_state)

    if (data_path / 'xgb_model.ubj').is_file():
        model = xgb.XGBClassifier()
        model.load_model(data_path / 'xgb_model.ubj')
    else:
        model = train_model(x_train, y_train, random_state, data_path)

    report, pred_confusion_matrix = test_model(model, decision_threshold, x_test, y_test)
    print(report)
    print(pred_confusion_matrix)

    print('done')
