import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import pandas as pd


def disease_prediction(df, config, json):
    ### 카테고리로 들어가있는 질병에 대해서 disease_list 칼럼을 새로 생성함
    df['disease_list'] = None

    for i in range(len(df)):
        df['disease_list'].iloc[i] = list(set(df.iloc[i, 83:93].to_list()) - {np.nan})

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
             'Urine protein', 'Urine Creatinine', 'pH(U)', 'SDMA', 'Na/K',
             'disease_list']]

    # 이름변경
    df.columns = ['no', 'test_no', 'pet_name', 'Hg', 'As', 'Cd', 'Pb', 'Al', 'Ba', 'U', 'Bi', 'Ni',
                  'Ti', 'Cs', 'Sb', 'Ca', 'Mg', 'Na', 'K', 'Cu', 'Zn', 'P', 'Fe', 'Mn',
                  'Cr', 'Se', 'Co', 'Li', 'V', 'Mo', 'B', 'Ca_Mg', 'Ca_P', 'Na_K',
                  'Zn_Cu', 'Na_Mg', 'Ca_K', 'cat_dog', 'dog_kind', 'age', 'sex', 'weight', 'BCS', 'skin',
                  'joint', 'heart', 'eye', 'respi', 'oral', 'diget', 'urinal', 'brain',
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
                  'Urine protein', 'Urine Creatinine', 'pH(U)', 'SDMA', 'Na_K(blood)',
                  'disease_list']

    # 성별의 미입력은 반드시 NULL로 입력
    df['sex'].fillna('NULL', inplace=True)

    # 그외 결측 -999로 보간
    df.fillna(-999, inplace=True)

    print(config['disease_list'])


    for disease in config['disease_list']:
        df[disease] = [int(1) if ((disease in cate_list) == True) else int(0) for cate_list in df['disease_list']]

    print(df['건강'])

    y_list = []
    for i in range(len(df)):
        y_list.append(df[config['disease_list']].iloc[i].to_list())
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


    # OHE 생성
    OHE = OneHotEncoder(handle_unknown='ignore')
    OHE.fit(X)
    X = OHE.transform(X)

    # Scaler 생성
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_split'],
                                                        random_state=config['split_seed'])
    # get the dataset
    classifier = MultiOutputClassifier(lgb.LGBMClassifier())

    model = Pipeline([('classify', classifier)])
    # print(model)

    model.fit(X_train, y_train)





    df_json = pd.DataFrame(json, index=['0'])

    X = OHE.transform(df_json)
    X = scaler.transform(X)

    result_dict = dict()
    for disease_name, disease_proba in zip(['health', 'liver', 'kidney', 'heart', 'tumor'], model.predict_proba(X)):
        result_dict[disease_name] = round(disease_proba[0][1]*100,2)


    return result_dict

