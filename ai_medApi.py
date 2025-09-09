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
import re

# NEW: OpenAI client (fallback)
from openai import AsyncOpenAI

# 🔑 Load environment variables
load_dotenv()

# API auth (comma-separated allowed keys)
API_KEYS = set(filter(None, [s.strip() for s in os.getenv("API_KEYS", "").split(",")]))
print("🔐 API_KEYS loaded:", bool(API_KEYS))

# ---- OpenAI ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ⚙️ MySQL configuration
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "Ai_Med")
DB_PORT = int(os.getenv("DB_PORT", 3306))

# Prefer a full DATABASE_URL if provided; else build one
DATABASE_URL = os.getenv("DATABASE_URL") or f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 🚀 Initialize FastAPI
app = FastAPI(
    title="Ai_Med API",
    description="Medical Assistant API with Doctor Recommendations and Explanations",
    version="1.4"
)

# 🌍 CORS Middleware (add more origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://shnetpaq.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 API Key Middleware
@app.middleware("http")
async def require_api_key(request: Request, call_next):
    # allow health checks without key
    if request.url.path in ("/", "/healthz"):
        return await call_next(request)
    api_key = request.headers.get("X-API-Key")
    if not API_KEYS or api_key not in API_KEYS:
        return JSONResponse(status_code=403, content={"detail": "Invalid or missing API Key"})
    return await call_next(request)

# 🌐 Translator & DB
translator = Translator()
db = databases.Database(DATABASE_URL)

# 📝 Optional Hunspell spell checker
try:
    import hunspell
    hunspell_checker = hunspell.HunSpell("./dictionaries/sq_AL.dic", "./dictionaries/sq_AL.aff")
    print("✅ Albanian dictionary loaded.")
except Exception as e:
    print(f"⚠️ Hunspell not available: {e}")
    hunspell_checker = None

# 🕒 Format time for DB fields
def format_time(value):
    if isinstance(value, timedelta):
        total_seconds = int(value.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours:02}:{minutes:02}"
    return str(value) if value is not None else None

# ⏱️ Async timing decorator
def timed_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return await func(*args, **kwargs)
        finally:
            print(f"⏱️ {func.__name__} took {time.time()-start:.3f}s")
    return wrapper

# 📦 Pydantic Models
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

class ExplanationResponse(BaseModel):
    advice: str  # frontend still expects "advice"

class RecommendedDoctorsResponse(BaseModel):
    doctors: List[Doctor]

# 🔌 Startup/Shutdown
@app.on_event("startup")
async def startup():
    await db.connect()
    # one shared HTTP client (used for Ollama and can be reused elsewhere)
    app.state.http_client = httpx.AsyncClient(timeout=60)
    print("🚀 Startup complete. DB connected.")

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()
    await app.state.http_client.aclose()
    print("🛑 Shutdown complete. DB disconnected.")

# 📝 Spell correction
def correct_albanian_text(text: str) -> str:
    if not hunspell_checker:
        return text
    corrected = []
    words = re.findall(r'\b\w+\b|[.,;!?]', text)
    for w in words:
        if re.match(r'\b\w+\b', w) and not hunspell_checker.spell(w):
            suggestions = hunspell_checker.suggest(w)
            corrected.append(suggestions[0] if suggestions else w)
        else:
            corrected.append(w)
    return ' '.join(corrected).replace(" ,", ",").replace(" .", ".")

async def correct_albanian_text_async(text: str) -> str:
    return await asyncio.to_thread(correct_albanian_text, text)

# 🌐 Translation with cache
@functools.lru_cache(maxsize=512)
def cached_translate(text: str, src: str, dest: str) -> str:
    return translator.translate(text, src=src, dest=dest).text

@timed_async
async def translate_text(text: str, source: str, target: str) -> str:
    return await asyncio.to_thread(cached_translate, text, source, target)

# 🧑‍⚕️ DB: doctors by specialty
@timed_async
async def fetch_doctors_by_specialty(specialization: str) -> List[Dict]:
    query = """SELECT name, email, is_admin, specialization, image, bio, quote, working_start, working_end, schooling
               FROM users WHERE LOWER(specialization) LIKE :specialization"""
    rows = await db.fetch_all(query=query, values={"specialization": f"%{specialization.lower()}%"})
    return [
        {
            "name": r["name"], "email": r["email"], "is_admin": r["is_admin"],
            "specialization": r["specialization"], "image": r["image"],
            "bio": r["bio"], "quote": r["quote"],
            "working_start": format_time(r["working_start"]),
            "working_end": format_time(r["working_end"]),
            "schooling": r["schooling"]
        } for r in rows
    ]

# 🤖 AI Model Call with Fallback
@timed_async
async def call_model_async(prompt: str) -> str:
    """
    Try Ollama first (localhost:11434). If unavailable or errors, fall back to OpenAI (gpt-4o-mini).
    Returns plain text. On total failure, returns a safe default.
    """
    # 1) Try Ollama (streaming)
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": "phi3:mini", "prompt": prompt, "stream": True}
        async with app.state.http_client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            text = ""
            async for line in resp.aiter_lines():
                if not line or not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text += data.get("response", "")
                except Exception:
                    continue
            if text.strip():
                return text.strip()
    except Exception as e:
        print(f"[Ollama unavailable, falling back to OpenAI] {e}")

    # 2) Fallback to OpenAI
    try:
        if not openai_client:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise medical triage assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or "general practitioner"
    except Exception as e:
        print(f"[OpenAI error] {e}")
        return "general practitioner"

# 🔮 Infer specialty
@timed_async
async def infer_specialty_async(symptoms: str) -> str:
    english = await translate_text(symptoms, "sq", "en")
    prompt = f'Patient symptoms: "{english}". Respond ONLY with the most appropriate medical specialist.'
    raw = await call_model_async(prompt)
    return raw.lower().strip()

# 💬 Explanation only
@timed_async
async def get_explanation_async(symptoms: str) -> str:
    english = await translate_text(symptoms, "sq", "en")
    prompt = f"""
You are an empathetic medical information assistant.
Patient symptoms: "{english}".
1. Start with a warm, empathetic phrase.
2. Mention 2–3 common general causes for the symptom.
3. End with: "This is for informational purposes only. Please consult a doctor."
"""
    en = await call_model_async(prompt)
    return await translate_text(en, "en", "sq")

# 🌐 Endpoints

@app.get("/")
async def root():
    return {"status": "ok", "name": "Ai_Med API", "version": "1.4"}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/infer-specialty", response_model=SpecialtyResponse, tags=["Inference"])
async def api_infer(req: SymptomsRequest):
    corrected = await correct_albanian_text_async(req.symptoms)
    spec = await infer_specialty_async(corrected)
    return {"symptom": corrected, "specialty": spec}

@app.post("/explanation", response_model=ExplanationResponse, tags=["Explanation"])
async def api_explanation(req: SymptomsRequest):
    corrected = await correct_albanian_text_async(req.symptoms)
    expl = await get_explanation_async(corrected)
    return {"advice": expl}  # still under "advice" key for frontend

@app.post("/recommend-doctors", response_model=RecommendedDoctorsResponse, tags=["Recommendation"])
async def api_recommend(req: SymptomsRequest):
    corrected = await correct_albanian_text_async(req.symptoms)
    spec_en = await infer_specialty_async(corrected)
    spec_sq = await translate_text(spec_en, "en", "sq")
    doctors = await fetch_doctors_by_specialty(spec_sq)
    if not doctors:
        rows = await db.fetch_all("SELECT DISTINCT specialization FROM users")
        available = ", ".join(sorted(set(r["specialization"] for r in rows)))
        raise HTTPException(404, f"No doctors found for '{spec_sq}'. Available: {available}")
    return {"doctors": doctors[:3]}
