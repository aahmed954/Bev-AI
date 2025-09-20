from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "autonomous-coordinator", "autonomous": True}

@app.get("/")
def root():
    return {"service": "autonomous-coordinator", "phase": 9}
