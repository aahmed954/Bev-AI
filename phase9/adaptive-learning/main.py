from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "adaptive-learning", "autonomous": True}

@app.get("/")
def root():
    return {"service": "adaptive-learning", "phase": 9}
