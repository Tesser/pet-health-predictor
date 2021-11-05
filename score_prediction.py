import numpy as np
from scipy import sparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def score_prediction(df, config, json):

    # 필요칼럼만 선별
    df = df[['순', '검체명', '동물명', 'Hg', 'As', 'Cd', 'Pb', 'Al', 'Ba', 'U', 'Bi', 'Ni',
             'Ti', 'Cs', 'Sb', 'Ca', 'Mg', 'Na', 'K', 'Cu', 'Zn', 'P', 'Fe', 'Mn',
             'Cr', 'Se', 'Co', 'Li', 'V', 'Mo', 'B', 'Ca_Mg', 'Ca_P', 'Na_K',
             'Zn_Cu', 'Na_Mg', 'Ca_K', '종', '품종', '나이', '성별', '몸무게', 'BCS', '피부건강',
             '관절건강', '심장건강', '눈건강', '호흡기건강', '구강건강', '소화기건강', '비뇨생식기건강', '뇌신경건강',
             '호르몬건강',
             '카테고리1', '카테고리2', '카테고리3', '카테고리4', '카테고리5', '카테고리6', '카테고리7',
             '카테고리8', '카테고리9', '카테고리10', '혈액검사일', 'Purpose_of_visit', 'Record',
             'Glucose', 'BUN', 'Creatinine', 'BUN/CREratio', 'ALP', 'ALT',
             'T_protein', 'Albumin', 'Globulin', 'A/G ratio', 'T_bilirubin', 'GGT',
             'T_cholesterol', 'Phosphorus', 'Calcium[Ca++]', 'Amylase',
             'Lipase(Dry)', 'Lipase', 'AST/GOT', 'ALT/GPT', 'TG', 'RBC', 'HCT',
             'HGB', 'MCV', 'MCH', 'MCHC', 'RDW-CV', '#RETIC', '%RETIC', 'RETHGB',
             'WBC', '%NEU', '%LYM', '%MON', 'EO(%)', '%BASO', '%GRA', '#NEU', '#LYM',
             '#MON', 'EO(#)', '#BASO', '#GRA', 'PLT', 'MPV', 'PDW', 'THT', 'cPL',
             'Na+', 'K+', 'Ca++', 'Cl-', 'pH', 'PCO2', 'PO2', 'HCO3-', 'BEecf(Ven)',
             'tCO2', 'ANION GAP', 'SO2', 'stO2', 'Hct(Ven)', 'CRP', 'UPC',
             'Urine protein', 'Urine Creatinine', 'pH(U)', 'SDMA', 'Na/K']]

    # 이름변경
    df.columns = ['no', 'test_no', 'pet_name', 'Hg', 'As', 'Cd', 'Pb', 'Al', 'Ba', 'U', 'Bi', 'Ni',
                  'Ti', 'Cs', 'Sb', 'Ca', 'Mg', 'Na', 'K', 'Cu', 'Zn', 'P', 'Fe', 'Mn',
                  'Cr', 'Se', 'Co', 'Li', 'V', 'Mo', 'B', 'Ca_Mg', 'Ca_P', 'Na_K',
                  'Zn_Cu', 'Na_Mg', 'Ca_K', 'cat_dog', 'dog_kind', 'age', 'sex', 'weight', 'BCS', 'skin',
                  'joint', 'heart', 'eye', 'respi', 'oral', 'diget', 'urin', 'brain',
                  'hormon', 'disease_1', 'disease_2', 'disease_3', 'disease_4', 'disease_5', 'disease_6', 'disease_7',
                  'disease_8', 'disease_9', 'disease_10', 'date', 'Purpose_of_visit', 'Record',
                  'Glucose', 'BUN', 'Creatinine', 'BUN_CREratio', 'ALP', 'ALT',
                  'T_protein', 'Albumin', 'Globulin', 'A_G ratio', 'T_bilirubin', 'GGT',
                  'T_cholesterol', 'Phosphorus', 'Calcium', 'Amylase',
                  'Lipase(Dry)', 'Lipase', 'AST_GOT', 'ALT_GPT', 'TG', 'RBC', 'HCT',
                  'HGB', 'MCV', 'MCH', 'MCHC', 'RDW-CV', 'shap_RETIC', 'percent_RETIC', 'RETHGB',
                  'WBC', 'percent_NEU', 'percent_LYM', 'percent_MON', 'percent_EO', 'percent_BASO', 'percent_GRA',
                  'shap_NEU', 'shap_LYM',
                  'shap_MON', 'shap_EO', 'shap_BASO', 'shap_GRA', 'PLT', 'MPV', 'PDW', 'THT', 'cPL',
                  'Na+', 'K+', 'Ca++', 'Cl-', 'pH', 'PCO2', 'PO2', 'HCO3-', 'BEecf(Ven)',
                  'tCO2', 'ANION GAP', 'SO2', 'stO2', 'Hct(Ven)', 'CRP', 'UPC',
                  'Urine protein', 'Urine Creatinine', 'pH(U)', 'SDMA', 'Na_K(blood)']

    # 성별의 미입력은 반드시 NULL로 입력
    df['sex'].fillna('NULL', inplace=True)

    # 그외 결측 -999로 보간
    df.fillna(-999, inplace=True)


    # y에 대한 멀티아웃풋 학습 레이블 생성
    y_list = []
    for i in range(len(df)):
        y_list.append(df[config['score_list']].iloc[i].to_list())
    y = np.array(y_list)
    y_sparse = sparse.csr_matrix(y)

    X = df[[

        # 중금속
        'Hg', 'As', 'Cd', 'Pb', 'Al', 'Ba', 'U', 'Bi', 'Ni',
        'Ti', 'Cs', 'Sb',

        # 영양 미네랄
        'Ca', 'Mg', 'Na', 'K', 'Cu', 'Zn', 'P', 'Fe', 'Mn',
        'Cr', 'Se', 'Co', 'Li', 'V', 'Mo', 'B',

        # 미네랄 비율
        'Ca_Mg', 'Ca_P', 'Na_K',
        'Zn_Cu', 'Na_Mg', 'Ca_K',

        # 기본 정보
        'age', 'sex', 'weight', 'BCS',

        # 혈액검사(일반)
        'Glucose', 'BUN', 'Creatinine', 'BUN_CREratio', 'ALP', 'ALT',
        'T_protein', 'Albumin', 'Globulin', 'A_G ratio', 'T_bilirubin', 'GGT',
        'T_cholesterol', 'Phosphorus', 'Calcium',
        'AST_GOT', 'ALT_GPT', 'TG',

        # 혈액검사(CBC)
        'RBC', 'HCT',
        'HGB', 'MCV', 'MCH', 'MCHC', 'RDW-CV', 'shap_RETIC', 'percent_RETIC', 'RETHGB',
        'WBC', 'percent_NEU', 'percent_LYM', 'percent_MON', 'percent_EO', 'percent_BASO', 'percent_GRA', 'shap_NEU',
        'shap_LYM',
        'shap_MON', 'shap_EO', 'shap_BASO', 'shap_GRA', 'PLT', 'MPV', 'PDW', 'THT',

        # 혈액가스(정맥혈)
        'Na+', 'K+', 'Ca++', 'Cl-', 'pH', 'PCO2', 'PO2', 'HCO3-', 'BEecf(Ven)',
        'tCO2', 'SO2', 'stO2', 'Hct(Ven)',

        #             #기타
        #             'CRP', 'UPC','ANION GAP', 'cPL',
        #        'Urine protein', 'Urine Creatinine', 'pH(U)', 'SDMA', 'Na_K','Amylase','Lipase(Dry)', 'Lipase'

    ]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_split'],
                                                        random_state=config['split_seed'])

    ### 1. X에 대한 원핫 인코딩
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    onehot_encoder.fit(X_train)

    ########################### 1-1. X에 대한 원핫인코딩 저장

    X_train = onehot_encoder.transform(X_train)
    X_test = onehot_encoder.transform(X_test)
    # ordinal encode target variable



    #### 2. y에 대한 원핫 인코딩
    label_encoder = LabelEncoder()

    encoded_y_list = []

    for i in range(10):
        label_encoder.fit(y_train[:, i])
        encoded_y_list.append(label_encoder.transform(y_train[:, i]))

    df_encoded_y = pd.DataFrame(encoded_y_list).T

    y_train = df_encoded_y.to_numpy()

    encoded_y_list = []

    for i in range(10):
        label_encoder.fit(y_train[:, i])
        encoded_y_list.append(label_encoder.transform(y_train[:, i]))

    df_encoded_y = pd.DataFrame(encoded_y_list).T
    df_encoded_y = df_encoded_y.to_numpy()

    y_train = df_encoded_y

    encoded_y_test_list = []

    for i in range(10):
        label_encoder.fit(y_test[:, i])
        encoded_y_test_list.append(label_encoder.transform(y_test[:, i]))

    encoded_y_test_list = pd.DataFrame(encoded_y_test_list).T
    encoded_y_test_list = encoded_y_test_list.to_numpy()

    y_test = encoded_y_test_list

    # get the dataset
    classifier = MultiOutputClassifier(lgb.LGBMClassifier())

    model = Pipeline([('classify', classifier)])
    # print(model)

    model.fit(X_train, y_train)





    df_json = pd.DataFrame(json, index=['0'])

    X = onehot_encoder.transform(df_json)


    print(model.predict(X))
    print(model.predict(X)[0])
    print(model.predict(X)[0][1])

    result_dict = dict()
    for i in range(len(config['score_list'])):
        result_dict[config['score_list'][i]] = str(model.predict(X)[0][i])

    return result_dict

