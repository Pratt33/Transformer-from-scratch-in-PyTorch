from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .loader import translate, load_model

app = FastAPI(
    title="English → Marathi Translator",
    description="Transformer-based neural machine translation",
    version="1.0.0"
)

# pydantic models define and validate request/response shapes
# FastAPI automatically generates OpenAPI docs from these
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    source:      str
    translation: str

# load model at startup — not on first request
# this ensures the first user doesn't experience a slow cold start
@app.on_event("startup")
def startup_event():
    load_model()
    print("Model ready")

@app.get("/health")
def health():
    # standard health check endpoint — load balancers ping this
    return {"status": "ok"}

@app.post("/translate", response_model=TranslationResponse)
def translate_text(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    try:
        result = translate(request.text)
        return TranslationResponse(source=request.text, translation=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))