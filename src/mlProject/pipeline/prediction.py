import joblib
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        model_path = Path(r'D:\DataScience\Ml flow project\artifacts\model_trainer\model.joblib')
        self.model = joblib.load(model_path)

    def predict(self, data):  # ✔ includes 'self'
        return self.model.predict(data)

    
    