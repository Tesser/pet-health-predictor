# YSIET 반려동물 질환 및 건강지수 예측 모델

## 개요
- 반려동물의 기본 정보(나이, 몸무게, 성별, BCS), 모발검사 정보, 혈액검사 정보를 json 포맷으로 입력 받아, <br>
1)  4종의 질환(간질환, 신장질환, 심장질환, 종양) 및 건강군에 속할 확률 값을 리턴해주는 모델<br>
2)  10종의 건강지수(피부, 관절, 심장, 눈, 호흡기, 구강, 소화기, 비뇨생식기, 뇌/신경 건강, 호르몬 건강)을 1~5점 단위로 예측하는 모델 <br>
을 포함하고 있습니다.
- 두 가지 모델은 light gbm을 기반으로 한 multi-output classifier를 이용하였으며, 상세한 모델 성적은 보고서에 기재하였습니다.
- 학습을 위해서 standard scaler 및 one-hot-encoder를 활용하였으며, 입력받는 데이터 역시 동일한 스케일의 정규화 및 원-핫 인코딩되도록 설계되어 있습니다.


## 간단한 시작법

1. 사용 전 pip 업그레이드를 해주세요.(Optional)
```shell
$ pip install --upgrade pip
```
<br>

2. 필요한 모듈을 다운로드 받아주세요.
```shell
$ pip install -r requirements.txt
```
<br>

3. main.py를 실행해주세요.
```shell
$ python main.py
```
<br>
- 정상적으로 API가 실행되었다면, 다음과 같은 화면을 확인하실 수 있습니다.

```shell
$ [2021-11-10 14:58:03,051] DEBUG in api: hello world
hello world
 * Serving Flask app "api" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Restarting with windowsapi reloader
[2021-11-10 14:58:04,540] DEBUG in api: hello world
hello world
 * Debugger is active!
 * Debugger PIN: 739-503-272
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

4. test.html을 실행해주시고, 파일 선택 버튼을 클릭하신 뒤 test_input.json을 선택해주세요. 이후 파란색 예측 버튼을 클릭해주세요.
- 정상적으로 파일이 선택되고, 예측이 시작되면 잠시 뒤 결과가 화면에 반환됩니다.


