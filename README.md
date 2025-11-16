# Synthetic Data Generator (SDE Project)

This project implements the full stack Synthetic data generator, data generated using VAE or GAN. 

## Features
- Client–Server Architecture
- FastAPI backend
- GAN + VAE synthetic tabular data generator
- Loan Prediction Dataset preprocessing
- Frontend: HTML + CSS + JavaScript (AJAX)
- Real-time synthetic data preview

## How to Clone

git clone https://github.com/Ram-Mihir-Prakki/Synthetic-Data-Generator.git

or (github cli)

gh repo clone Ram-Mihir-Prakki/Synthetic-Data-Generator


## How to Run
1. Download the loan dataset from kaggle : https://www.kaggle.com/datasets/burak3ergun/loan-data-set
2. Clone the github repo (steps mentioned above)
3. Rename the downloaded dataset to "loan.csv" and paste it in data directory (already available if cloned)
4. Run the below command to install the complete requirements:
        pip install -r requirements.txt
5. Run the command to start the uvicorn server in your local machine:
        uvicorn backend.fastapi.main:app --reload
6. Visit http://127.0.0.1:8000/ where the complete frontend is available. 



## API
POST /api/generate  
Payload: { "model": "gan" | "vae", "rows": 5–30 }

## Future Work
- Real GAN/VAE training
- Save/load artifacts
- Privacy metrics
- Custom Datasets and feature to upload dataset
- Realtime data set preprocessing for custom datasets
- Save the results(implementation of Database)
- Authentication using OAuth2.0