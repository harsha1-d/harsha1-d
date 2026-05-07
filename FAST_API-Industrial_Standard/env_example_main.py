from fastapi import FastAPI
from env_example_config import setting

app = FastAPI()

@app.get("/info")
def get_info():
    return {
        "debug": setting.debug,
        "database_url": setting.database_url
    }