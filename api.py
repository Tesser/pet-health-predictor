import sys
import logging
from flask_cors import CORS
from flask import Flask, jsonify, request
from utils.preprocessing_json import *
from utils.prediction import *

app = Flask(__name__)
CORS(app)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)
app.logger.debug("hello world")


@app.route('/', methods=['GET'])
def index():
    app.logger.info(__name__)
    return 'server connected'


@app.route('/predict_score', methods=['GET', 'OPTIONS', 'POST'])
def score_predict():
    app.logger.info(__name__)
    feature = request.form.to_dict()  # json을 dict로 변환해서 파이썬에 입력으로 받음

    try:
        app.logger.info(feature)
        app.logger.info('prediction started!')
        df_json = preprocessing_json(feature)
        result_dict = score_prediction(df_json)

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
        df_json = preprocessing_json(feature)
        result_dict = disease_prediction(df_json)

    except Exception as e:
        app.logger.info(str(e))
        return jsonify({'error': 'Failed to predict:' + str(e)})

    return result_dict


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
