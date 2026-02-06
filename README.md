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
