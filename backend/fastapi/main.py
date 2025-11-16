from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import random

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


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/api/generate")
async def generate_date(request: Request):
    try:
        payload = await request.json()
        model = payload.get("model", "vae")
        rows = int(payload.get("rows", 5))
        if rows < 1:
            rows = 1
        if rows > 1000:
            rows = 1000

        columns = ["id", "age", "income", "category", "score"]
        rows_out = []
        rng = random.Random(42)
        for i in range(rows):
            rid = f"r{i+1:02d}"
            if model == "gan":
                age = 25 + (rng.randint(0, 40))
                income = 30000 + rng.randint(0, 90000)
            else:  
                age = 22 + (rng.randint(0, 45))
                income = 28000 + rng.randint(0, 85000)

            category = ["A", "B", "C"][rng.randint(0, 2)]
            score = f"{(rng.random() * 100):.2f}"
            rows_out.append([rid, age, income, category, score])

        return JSONResponse(content={"columns": columns, "rows": rows_out}) 

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
