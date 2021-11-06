from config import *
from score_prediction import *
from disease_prediction import *
import pandas as pd
import sys
import logging
from flask_cors import CORS
from flask import Flask, jsonify, request, render_template
from functions import *


df = pd.read_csv('./data/YSIET_ver03.csv', encoding='utf8')
port = 5000

app = Flask(__name__)
CORS(app)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)
app.logger.debug("hello world")


@app.route('/', methods=['GET'])
def index():
    app.logger.info(__name__)
    return 'server connected'

@app.route('/test', methods=['GET'])
def test():
    app.logger.info(__name__)
    return render_template('test.html', api='localhost:' + str(port))

@app.route('/predict_score', methods=['GET', 'OPTIONS', 'POST'])
def score_predict():
    app.logger.info(__name__)
    feature = request.form.to_dict()  # json을 dict로 변환해서 파이썬에 입력으로 받음

    try:
        app.logger.info(feature)
        app.logger.info('prediction started!')
        json = astype_json(feature)
        result_dict = score_prediction(df, config, json)
        print(result_dict)

    except Exception as e:
        app.logger.info(str(e))
        return jsonify({'error': 'Failed to predict:' + str(e)})

    return result_dict


@app.route('/predict_disease', methods=['GET', 'OPTIONS', 'POST'])
def disease_predict():
    app.logger.info(__name__)
    feature = request.form.to_dict()  # json을 dict로 변환해서 파이썬에 입력으로 받음

    try:
        app.logger.info(feature)
        app.logger.info('prediction started!')
        json = astype_json(feature)
        result_dict = disease_prediction(df, config, json)
        print(result_dict)

    except Exception as e:
        app.logger.info(str(e))
        return jsonify({'error': 'Failed to predict:' + str(e)})

    return result_dict


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)

