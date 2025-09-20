from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "resource-manager", "autonomous": True}

@app.get("/")
def root():
    return {"service": "resource-manager", "phase": 9}
