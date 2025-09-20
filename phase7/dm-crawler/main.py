from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "dm-crawler"}

@app.get("/")
def root():
    return {"service": "dm-crawler", "phase": 7}
