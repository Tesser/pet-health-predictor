import pandas as pd
import joblib
import pickle


def disease_prediction(json):

    scaler = joblib.load('./src/scaler/disease_scaler.save')
    OHE = joblib.load('./src/one_hot_encoder/disease_OHE.joblib')
    model = pickle.load(open('./src/model/disease_model.sav', 'rb'))

    df_json = pd.DataFrame(json, index=['0'])
    df_json.fillna(-999, inplace=True)

    X = OHE.transform(df_json)
    X = scaler.transform(X)

    disease_list = ['health', 'liver', 'kidney', 'heart', 'tumor']

    result_dict = dict()
    for disease_name, disease_proba in zip(disease_list, model.predict_proba(X)):
        result_dict[disease_name] = round(disease_proba[0][1]*100,2)

    return result_dict



def score_prediction(json):

    scaler = joblib.load('./src/scaler/score_scaler.save')
    OHE = joblib.load('./src/one_hot_encoder/score_OHE.joblib')
    model = pickle.load(open('./src/model/score_model.sav', 'rb'))

    df_json = pd.DataFrame(json, index=['0'])
    df_json.fillna(-999, inplace=True)

    X = OHE.transform(df_json)
    X = scaler.transform(X)

    score_list = ['index_skin', 'index_joint', 'index_heart', 'index_eye', 'index_respiratory', 'index_oral', 'index_digest', 'index_urinal', 'index_brain', 'index_hormone']

    result_dict = dict()
    for i in range(len(score_list)):
        result_dict[score_list[i]] = str(model.predict(X)[0][i])

    return result_dict
