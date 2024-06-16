# Discount detector application

This project is a FastAPI application that uses PyTorch, llama.cpp. Below are the instructions to run the application locally in a virtual environment and via Docker.

## Prerequisites

- Python 3.12
- pip
- Docker (optional, for Docker setup)



## Running with Docker

### Note for Windows Users
Docker can run on Windows if you have Docker Desktop installed. Make sure to enable WSL 2 (Windows Subsystem for Linux 2) during the installation for better performance and compatibility.

### Step 1: Build Docker Image

Pull the Docker image from Docker Hub repository:

```shell
docker pull klordoo/discount-detector:latest
```

### Step 2: Run Docker Container
Run the Docker container:

```shell
docker run -d -p 8000:8000 klordoo/discount-detector:latest
```
The application will be available at `http://localhost:8000`.

### Step 3: Access API Documentation

## You can access the automatically generated API documentation at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Additional Note
- Ensure you have the necessary permissions to run Docker commands.



## Running Locally

### Step 1: Clone the Repository
```shell
git clone https://github.com/e0xextazy/lct-2024-ner.git
```
```shell
cd lct-2024-ner
```

### Step 2: Create and Activate Virtual Environment

1. Create a virtual environment:
```shell
python -m venv venv
```
2. Activate the virtual environment:
```shell
source venv/bin/activate
```

### Step 3: Install Dependencies

## Install the required Python packages:
```shell
cd service
```
```shell
pip install -r requirements.txt
```

### Step 3: Run the Application

## Start the FastAPI application using Uvicorn:
```shell
uvicorn main:app
```

#### The application will be available at `http://127.0.0.1:8000`.

### Step 4: Access API Documentation

## You can access the automatically generated API documentation at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Documentation

The FastAPI application automatically generates interactive API documentation with Swagger UI and ReDoc.

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Endpoints

- **Get Predictions**
  - **URL**: `/predict`
  - **Method**: `POST`
  - **Request Body**:
    ```json
    {
      "text": "Sample text to process"
    }
    ```
  - **Response**:
    ```json
    [
      "O",
      "O",
      "O",
      "O"
    ]
    ```

### Prediction Types

The `/predict` endpoint returns a list of prediction types, which can be one of the following values:

- `O`
- `B_discount`
- `B_value`
- `I_value`

For more details, please refer to the interactive API documentation linked above.