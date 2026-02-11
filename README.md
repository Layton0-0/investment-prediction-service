# investment-prediction-service

AI 예측 및 스코어링 전용 마이크로서비스 (FastAPI, Stateless).

- **역할**: LSTM/ML 추론. Backend는 API 클라이언트로만 접근. 장애 시 Backend는 graceful fallback.
- **수평 확장 가능.**

## 구조

```
├── app/
│   ├── main.py
│   └── models/
├── Dockerfile
├── requirements.txt
├── Jenkinsfile
└── README.md
```

## 테스트

의존성 설치 후 API·전처리·모델 테스트 실행:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

(Windows에서 `pip`가 PATH에 없으면 `python -m pip` 사용.)

- `tests/test_api.py`: GET /api/v1/health, POST /api/v1/predict, POST /api/v1/predict/batch
- `tests/test_preprocessing.py`, `tests/test_lstm_model.py`: 전처리·LSTM 단위 테스트

## 보안·운영

- **CORS**: 현재 개발 편의를 위해 `allow_origins=["*"]`로 설정되어 있음. **프로덕션**에서는 반드시 허용할 오리진을 제한할 것 (예: Backend/프론트 도메인만). 환경 변수 `CORS_ORIGINS`가 설정되면 해당 값(쉼표 구분)을 사용하고, 미설정 시 기본값 `*` 사용.
- 이 서비스는 내부망에서만 Backend가 호출하는 구조를 권장하며, 외부 노출 시 인증·Rate limit 등을 고려할 것.
