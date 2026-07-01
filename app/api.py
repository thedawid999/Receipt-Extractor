from fastapi import FastAPI
from .utils import load_config
from .scheduler import start_scheduler, stop_scheduler, daily_job
from pydantic import BaseModel

app = FastAPI()

# to make sure datatypes are correct
class ScheduleRequest(BaseModel):
    input_dir: str
    output_dir: str
    output_file: str = "results.json"
    hour: int
    minute: int

class SingleRequest(BaseModel):
    input_dir: str
    output_dir: str
    output_file: str = "results.json"

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/process")
def process(req: SingleRequest):
    config["input_dir"] = req.input_dir
    config["output_dir"] = req.output_dir
    config["output_file"] = req.output_file

    return daily_job(config)

@app.post("/scheduler/start")
def start_schedule(req: ScheduleRequest):
    config["schedule"]["hour"] = req.hour
    config["schedule"]["minute"] = req.minute

    start_scheduler(config)
    return {"status": "scheduler started"}

@app.post("/scheduler/stop")
def stop_schedule():
    stop_scheduler()
    return {"status": "stopped"}

@app.get("/config")
def get_config():
    return config