from fastapi import FastAPI
from app.controller.hgt_controller import router as hgt_router

def create_app() -> FastAPI:
    app = FastAPI(title="HGT-Autoscaler", version="1.0.0")
    app.include_router(hgt_router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
