from locust import HttpUser, task


class HelloWorldUser(HttpUser):
    @task
    def test_predict(self):
        self.client.post("/predict", json={'text': 'Test text'})

    def on_start(self):
        response = self.client.post("/predict", json={'text': 'Text 1'})
        print(response.status_code, response.json())

        response = self.client.post("/predict", json={'text': 'Text 2'})
        print(response.status_code, response.json())

        response = self.client.post("/predict", json={'text': 'Text 3'})
        print(response.status_code, response.json())
