# fasterrcnn-mlops-pipeline
Object Detection Pipeline for Model Production and Deployment

fasterrcnn-mlops-pipeline/
├── app/                        # FastAPI app
│   ├── main.py
│   ├── api/
│   │   └── endpoints.py
│   ├── schemas/
│   │   └── request_response.py
│   └── services/
│       └── inference.py
├── model/                      # Model training, evaluation
│   ├── data/
│   ├── training.py
│   ├── model.py
│   └── config.py
├── tests/                      # Pytest unit/integration tests
│   ├── test_inference.py
│   └── test_api.py
├── docker/                     # Docker & Kubernetes config
│   ├── Dockerfile
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   └── service.yaml
├── .github/workflows/          # CI/CD with GitHub Actions
│   └── ci.yml
├── scripts/                    # Utility or preprocessing scripts
│   └── setup_data.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── .env.example

lsof -ti tcp:5000 | xargs kill -9


pip install -r requirements.txt

mlflow ui
open: http://127.0.0.1:5000

python -m model.training
uvicorn app.main:app --reload
open: http://127.0.0.1:8000/docs
pytest

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
python test_api.py




# .github/workflows/ci.yml
"""
name: CI

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest
    - name: Check lint
      run: |
        black --check .
        flake8
        mypy .
"""

# .env.example
"""
MODEL_PATH=model/fasterrcnn.pth
"""