from apscheduler.schedulers.background import BackgroundScheduler
from .layoutlm import predict
from .utils import save_to_file, resolve_image_paths, load_config
from pathlib import Path
import os
import time

api_mode = os.getenv("API_MODE", "false").lower() == "true"
scheduler = BackgroundScheduler()

def start_scheduler(config):
    if api_mode:
        start_scheduler_api(config)
    else:
        start_scheduler_cli(config)

# cli version contains infinite loop, which cannot be inside of api version
def start_scheduler_cli(config):
    scheduler.add_job(
        daily_job,
        trigger="cron",
        hour=config["schedule"]["hour"],
        minute=config["schedule"]["minute"],
        args=[config]
    )

    print(f"Scheduler started ({config['schedule']['hour']:02d}:{config['schedule']['minute']:02d})")
    scheduler.start()

    try:
        while True:
            time.sleep(1)   
    except KeyboardInterrupt:
        print("Stopping scheduler...")
        scheduler.shutdown()

# api version runs without the infinite loop
def start_scheduler_api(config):
    scheduler.add_job(
        daily_job,
        trigger="cron",
        hour=config["schedule"]["hour"],
        minute=config["schedule"]["minute"],
        args=[config]
    )

    print(f"Scheduler started ({config['schedule']['hour']:02d}:{config['schedule']['minute']:02d})")
    scheduler.start()

# contains every step that will be done once the scheduler starts
def daily_job(config):
    input_dir = config["input_dir"]
    output_dir = Path(config["output_dir"])
    output_file = config["output_file"]

    results = []

    for path in resolve_image_paths(input_dir):
        results.append(predict(path))

    output_dir.mkdir(exist_ok=True)
    save_to_file(output_dir / output_file, results)
    print("Daily job finished!")

def stop_scheduler():
    scheduler.shutdown()



