"""
API 엔드포인트 수준 테스트 (FastAPI TestClient).
GET /api/v1/health, POST /api/v1/predict, POST /api/v1/predict/batch 검증.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHealthApi(unittest.TestCase):
    """GET /api/v1/health"""

    def test_health_returns_200_and_ok(self):
        response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("service", data)
        self.assertIn("timestamp", data)


class TestPredictApi(unittest.TestCase):
    """POST /api/v1/predict"""

    def test_predict_minimal_body_returns_200_and_prediction(self):
        body = {
            "symbol": "005930",
            "predictionMinutes": 60,
        }
        response = client.post("/api/v1/predict", json=body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("symbol"), "005930")
        self.assertIn("currentPrice", data)
        self.assertIn("predictedPrice", data)
        self.assertIn("confidence", data)
        self.assertIn("direction", data)
        self.assertIn("modelType", data)
        self.assertEqual(data.get("predictionMinutes"), 60)

    def test_predict_with_optional_fields_returns_200(self):
        body = {
            "symbol": "AAPL",
            "predictionMinutes": 120,
            "modelType": "ensemble",
            "lookbackDays": 30,
        }
        response = client.post("/api/v1/predict", json=body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("symbol"), "AAPL")
        self.assertEqual(data.get("predictionMinutes"), 120)


class TestPredictBatchApi(unittest.TestCase):
    """POST /api/v1/predict/batch"""

    def test_predict_batch_returns_200_and_list(self):
        body = [
            {"symbol": "005930", "predictionMinutes": 60},
            {"symbol": "000660", "predictionMinutes": 30},
        ]
        response = client.post("/api/v1/predict/batch", json=body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0].get("symbol"), "005930")
        self.assertEqual(data[1].get("symbol"), "000660")


if __name__ == "__main__":
    unittest.main()
