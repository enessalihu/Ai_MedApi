from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import json
import httpx
import asyncio
import time
import functools
from googletrans import Translator
from dotenv import load_dotenv
import os
import databases
from datetime import timedelta
from fastapi.middleware.cors import CORSMiddleware

# Ngarkon variablat e mjedisit
load_dotenv()

# API Keys
API_KEYS = set(os.getenv("API_KEYS", "").split(","))

# MySQL konfigurimi
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "Ai_Med")
DB_PORT = int(os.getenv("DB_PORT", 3306))

DATABASE_URL = f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Inicioni aplikacionin
app = FastAPI(
    title="Ai_Med API",
    description="Medical Assistant API with Doctor Recommendations and Advice",
    version="1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.shnetpaq.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware për API Key
@app.middleware("http")
async def require_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if api_key not in API_KEYS:
        return JSONResponse(status_code=403, content={"detail": "Invalid or missing API Key"})
    return await call_next(request)

translator = Translator()
db = databases.Database(DATABASE_URL)

# Funksion për të konvertuar timedelta në string
def format_time(value):
    if isinstance(value, timedelta):
        total_seconds = int(value.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours:02}:{minutes:02}"
    return str(value) if value is not None else None

# Dekorator për matje kohe
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

# Modelet Pydantic
class Doctor(BaseModel):
    name: str
    email: str
    is_admin: bool
    specialization: str
    image: str
    bio: str
    quote: str
    working_start: str
    working_end: str
    schooling: str

class SymptomsRequest(BaseModel):
    symptoms: str

class SpecialtyResponse(BaseModel):
    symptom: str
    specialty: str

class AdviceResponse(BaseModel):
    advice: str

class RecommendedDoctorsResponse(BaseModel):
    doctors: List[Doctor]

# Ngarkimi i DB dhe klientit HTTP
@app.on_event("startup")
async def startup():
    await db.connect()
    app.state.http_client = httpx.AsyncClient(timeout=60)

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()
    await app.state.http_client.aclose()

# Përkthimi me cache
@functools.lru_cache(maxsize=512)
def cached_translate(text: str, src: str, dest: str) -> str:
    return translator.translate(text, src=src, dest=dest).text

@timed_async
async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    return await asyncio.to_thread(cached_translate, text, source_lang, target_lang)

# Merr mjekët sipas specializimit
@timed_async
async def fetch_doctors_by_specialty(specialization: str) -> List[Dict]:
    query = """
        SELECT name, email, is_admin, specialization, image, bio, quote, working_start, working_end, schooling
        FROM users
        WHERE LOWER(specialization) LIKE :specialization
    """
    rows = await db.fetch_all(query=query, values={"specialization": f"%{specialization.lower()}%"})
    doctors = []
    for row in rows:
        doctors.append({
            "name": row["name"],
            "email": row["email"],
            "is_admin": row["is_admin"],
            "specialization": row["specialization"],
            "image": row["image"],
            "bio": row["bio"],
            "quote": row["quote"],
            "working_start": format_time(row["working_start"]),
            "working_end": format_time(row["working_end"]),
            "schooling": row["schooling"]
        })
    return doctors

# Thirr modelin AI për gjenerim përgjigjeje
@timed_async
async def call_model_async(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": "phi3:mini", "prompt": prompt, "stream": True}

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

# Inferenca e specializimit
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

# Merr këshillë mjekësore
@timed_async
async def get_advice_async(symptoms: str) -> str:
    english_symptoms = await translate_text(symptoms, source_lang="sq", target_lang="en")
    prompt = f"""You are a medical assistant. Based on the symptoms: "{english_symptoms}", 
respond with only one short sentence of first-step medical advice the patient can do on their own.
Do not mention doctors, hospitals, or professionals."""
    advice_en = await call_model_async(prompt)
    advice_sq = await translate_text(advice_en, source_lang="en", target_lang="sq")
    return advice_sq

# Endpointet
@app.get("/doctors", response_model=List[Doctor], tags=["Doctors"])
@timed_async
async def api_fetch_doctors():
    query = """
        SELECT name, email, is_admin, specialization, image, bio, quote, working_start, working_end, schooling
        FROM users
    """
    rows = await db.fetch_all(query)
    if not rows:
        raise HTTPException(status_code=404, detail="No doctors found")
    return [
        {
            "name": row["name"],
            "email": row["email"],
            "is_admin": row["is_admin"],
            "specialization": row["specialization"],
            "image": row["image"],
            "bio": row["bio"],
            "quote": row["quote"],
            "working_start": format_time(row["working_start"]),
            "working_end": format_time(row["working_end"]),
            "schooling": row["schooling"]
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
    specialty_en = await infer_specialty_async(req.symptoms)
    specialty_al = await translate_text(specialty_en, source_lang="en", target_lang="sq")
    doctors = await fetch_doctors_by_specialty(specialty_al)
    if not doctors:
        query = "SELECT DISTINCT specialization FROM users"
        rows = await db.fetch_all(query)
        available_specializations = ", ".join(set(row["specialization"] for row in rows))
        raise HTTPException(
            status_code=404,
            detail=f"No doctors found for specialization '{specialty_al}'. Available: {available_specializations}"
        )
    return {"doctors": doctors[:3]}
