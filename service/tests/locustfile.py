from locust import HttpUser, task
import pandas as pd
from tqdm import tqdm

texts = pd.read_csv('../../data/gt_test.csv')['processed_text']


class HelloWorldUser(HttpUser):
    @task
    def test_predict(self):
        for text in tqdm(texts):
            response = self.client.post("/predict", json={'text': text})
            response.raise_for_status()
        print('OK')

    def on_start(self):
        response = self.client.post("/predict", json={'text': 'Text 1'})
        print(response.status_code, response.json())

        response = self.client.post("/predict", json={'text': 'Text 2'})
        print(response.status_code, response.json())

        response = self.client.post("/predict", json={'text': 'Text 3'})
        print(response.status_code, response.json())
