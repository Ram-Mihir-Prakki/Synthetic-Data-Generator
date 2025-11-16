import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from backend.models.processing import Preprocessor
from backend.models.gan import GANModel
from backend.models.vae import VAEModel
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
templates = Jinja2Templates(directory="ui/templates")
pre = Preprocessor()
dataset_path = os.getenv("DATASET_CSV", "")
if dataset_path and Path(dataset_path).exists():
    try:
        pre.fit_from_csv(dataset_path)
    except Exception:
        pre.fit_from_dataframe(pd.DataFrame({c: [0, 1] for c in pre.default_num_cols + pre.default_cat_cols}))
else:
    try:
        if Path.cwd().joinpath("data", "loan.csv").exists():
            pre.fit_from_csv(str(Path.cwd().joinpath("data", "loan.csv")))
        else:
            pre.fit_from_dataframe(pd.DataFrame({c: [0, 1] for c in pre.default_num_cols + pre.default_cat_cols}))
    except Exception:
        pre.fit_from_dataframe(pd.DataFrame({c: [0, 1] for c in pre.default_num_cols + pre.default_cat_cols}))
gan_model = GANModel(pre)
vae_model = VAEModel(pre)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/api/generate")
async def api_generate(request: Request):
    try:
        payload = await request.json()
        model_name = payload.get("model", "gan")
        rows = int(payload.get("rows", 5))
        rows = max(1, min(rows, 1000))
        if model_name == "vae":
            rows_out = vae_model.sample(rows)
        else:
            rows_out = gan_model.sample(rows)
        columns = ["id"] + pre.columns()
        return JSONResponse(content={"columns": columns, "rows": rows_out})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
