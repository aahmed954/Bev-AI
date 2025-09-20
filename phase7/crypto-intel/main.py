from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "crypto-intel"}

@app.get("/")
def root():
    return {"service": "crypto-intel", "phase": 7}
