from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "economics-processor"}

@app.get("/")
def root():
    return {"service": "economics-processor", "phase": 7}
