# California Housing Price Prediction

A machine learning application that predicts California house prices using a **Random Forest Regressor**. The model is served through a **FastAPI REST API**, containerized using **Docker**, and deployed on **Render**.

## Tech Stack

* Python
* Scikit-learn
* FastAPI
* Docker
* Git & GitHub
* Render

## Project Structure

```text
ML_in_Production_assignment/
├── main.py
├── california_housing_model.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── static/
    └── index.html
```

## Machine Learning Model

The California Housing dataset is used to train a `RandomForestRegressor`.

The model uses 8 input features:

* MedInc
* HouseAge
* AveRooms
* AveBedrms
* Population
* AveOccup
* Latitude
* Longitude

To keep the model lightweight, a smaller portion of the dataset and a smaller Random Forest configuration were used.

## FastAPI

FastAPI provides the prediction API.

### Endpoints

`GET /`

Loads the web interface.

`GET /health`

Checks whether the API is running.

`POST /predict`

Accepts housing features and returns the predicted house price.

Example response:

```json
{
  "prediction": 2.85,
  "prediction_dollars": "$285,000.00"
}
```

## Web Interface

The application provides sliders for all 8 input features. After entering the values, the browser sends the data to the `/predict` endpoint and displays the predicted house price.

## Docker

The application is containerized using Docker.

Build the image:

```bash
docker build -t california-housing-api .
```

Run the container:

```bash
docker run -p 8000:8000 california-housing-api
```

The application is then available at:

```text
http://localhost:8000
```

## GitHub

The source code, Docker configuration, FastAPI application, frontend, and trained model are maintained in the GitHub repository.

## Render Deployment

After pushing the project to GitHub, the repository was connected to **Render** for deployment.

Render builds the Docker image and runs the FastAPI application inside the container.

The application is deployed as a web service and can be accessed through the public Render URL.

## Workflow

```text
California Housing Dataset
          ↓
   Train ML Model
          ↓
  Save .pkl Model
          ↓
       FastAPI
          ↓
     Web Interface
          ↓
       Docker
          ↓
       GitHub
          ↓
       Render
          ↓
   Public Web Application
```

## Conclusion

This project demonstrates a complete machine learning deployment workflow, from model training and API development to Docker containerization and cloud deployment using Render.
