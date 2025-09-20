from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "reputation-analyzer"}

@app.get("/")
def root():
    return {"service": "reputation-analyzer", "phase": 7}
