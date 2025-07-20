from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import json
import asyncpg
import httpx
import asyncio
import time
import functools
from googletrans import Translator
from dotenv import load_dotenv
import os

# Ngarko .env
load_dotenv()

# Merr API Keys (mund të jenë disa, të ndara me presje)
API_KEYS = set(os.getenv("API_KEYS", "").split(","))

# Merr kredencialet e databazës nga .env
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "Ai_Med")
DB_MIN_POOL = int(os.getenv("DB_MIN_POOL", 5))
DB_MAX_POOL = int(os.getenv("DB_MAX_POOL", 20))

app = FastAPI(
    title="Ai_Med API",
    description="Medical Assistant API with Doctor Recommendations and Advice",
    version="1.0"
)

# Middleware për verifikimin e API Key
@app.middleware("http")
async def require_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if api_key not in API_KEYS:
        return JSONResponse(status_code=403, content={"detail": "Invalid or missing API Key"})
    return await call_next(request)

# Initialize Google Translator once
translator = Translator()

def timed_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            return result
        except Exception as e:
            end_time = time.time()
            print(f"{func.__name__} failed after {end_time - start_time:.4f} seconds with error: {e}")
            raise
    return wrapper

class Doctor(BaseModel):
    name: str
    specialty: str
    location: str
    rating: float
    available_days: List[str]

class SymptomsRequest(BaseModel):
    symptoms: str

class SpecialtyResponse(BaseModel):
    symptom: str
    specialty: str

class AdviceResponse(BaseModel):
    advice: str

class RecommendedDoctorsResponse(BaseModel):
    doctors: List[Doctor]

@app.on_event("startup")
async def startup():
    app.state.db_pool = await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        min_size=DB_MIN_POOL,
        max_size=DB_MAX_POOL,
    )
    app.state.http_client = httpx.AsyncClient(timeout=60)

@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()
    await app.state.http_client.aclose()

@functools.lru_cache(maxsize=512)
def cached_translate(text: str, src: str, dest: str) -> str:
    return translator.translate(text, src=src, dest=dest).text

@timed_async
async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    return await asyncio.to_thread(cached_translate, text, source_lang, target_lang)

@timed_async
async def fetch_doctors_by_specialty(specialty: str) -> List[Dict]:
    async with app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT name, specialty, location, rating, available_days FROM doctors WHERE LOWER(specialty) LIKE $1",
            f"%{specialty.lower()}%"
        )
        doctors = []
        for row in rows:
            doctors.append({
                "name": row["name"],
                "specialty": row["specialty"],
                "location": row["location"],
                "rating": float(row["rating"]),
                "available_days": json.loads(row["available_days"]) if isinstance(row["available_days"], str) else row["available_days"]
            })
        return doctors

@timed_async
async def call_model_async(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi3:mini",
        "prompt": prompt,
        "stream": True
    }

    try:
        async with app.state.http_client.stream("POST", url, json=payload, timeout=60) as response:
            response.raise_for_status()
            generated_text = ""

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    generated_text += token
                except json.JSONDecodeError as e:
                    print(f"[streaming] JSON decode error: {e}")

            return generated_text.strip() or "general practitioner"

    except Exception as e:
        print(f"[call_model_async - stream] Error: {e}")
        return "general practitioner"

@timed_async
async def infer_specialty_async(symptoms: str) -> str:
    english_symptoms = await translate_text(symptoms, source_lang="sq", target_lang="en")

    prompt = f"""Patient symptoms: \"{english_symptoms}\".

Choose the MOST appropriate medical specialist from the list below.
Respond ONLY with the exact name, no explanation, no punctuation, no context."""
    raw_output = await call_model_async(prompt)

    try:
        parsed = json.loads(raw_output)
        specialty_en = list(parsed.values())[0].lower()
    except Exception:
        specialty_en = raw_output.lower()

    return specialty_en

@timed_async
async def get_advice_async(symptoms: str) -> str:
    english_symptoms = await translate_text(symptoms, source_lang="sq", target_lang="en")
    prompt = f"""You are a medical assistant. Based on the symptoms: "{english_symptoms}", 
respond with only one short sentence of first-step medical advice the patient can do on their own.
Do not mention doctors, hospitals, or professionals."""
    advice_en = await call_model_async(prompt)
    advice_sq = await translate_text(advice_en, source_lang="en", target_lang="sq")
    return advice_sq

def find_doctors(specialty: str, all_doctors: List[Dict]) -> List[Dict]:
    specialty_lower = specialty.lower()
    return [d for d in all_doctors if specialty_lower in d["specialty"].lower()]

@app.get("/doctors", response_model=List[Doctor], tags=["Doctors"])
@timed_async
async def api_fetch_doctors():
    async with app.state.db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT name, specialty, location, rating, available_days FROM doctors")
        if not rows:
            raise HTTPException(status_code=404, detail="No doctors found")
        return [
            {
                "name": row["name"],
                "specialty": row["specialty"],
                "location": row["location"],
                "rating": float(row["rating"]),
                "available_days": json.loads(row["available_days"]) if isinstance(row["available_days"], str) else row["available_days"]
            }
            for row in rows
        ]

@app.post("/infer-specialty", response_model=SpecialtyResponse, tags=["Inference"])
@timed_async
async def api_infer_specialty(req: SymptomsRequest):
    specialty = await infer_specialty_async(req.symptoms)
    return {"symptom": req.symptoms, "specialty": specialty}

@app.post("/advice", response_model=AdviceResponse, tags=["Advice"])
@timed_async
async def api_get_advice(req: SymptomsRequest):
    advice = await get_advice_async(req.symptoms)
    return {"advice": advice}

@app.post("/recommend-doctors", response_model=RecommendedDoctorsResponse, tags=["Recommendation"])
@timed_async
async def api_recommend_doctors(req: SymptomsRequest):
    specialty = await infer_specialty_async(req.symptoms)
    doctors = await fetch_doctors_by_specialty(specialty)

    if not doctors:
        async with app.state.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT DISTINCT specialty FROM doctors")
            available_specialties = ", ".join(set(row["specialty"] for row in rows))
        raise HTTPException(
            status_code=404,
            detail=f"No doctors found for specialty '{specialty}'. Available: {available_specialties}"
        )

    top_doctors = sorted(doctors, key=lambda d: d['rating'], reverse=True)[:3]
    return {"doctors": top_doctors}
