from apscheduler.schedulers.blocking import BlockingScheduler
from layoutlm import predict
from utils import save_to_file, resolve_image_paths, load_config

# contains every step that will be done once the scheduler starts
def daily_job():
    config = load_config()

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    output_file = config["output_file"]

    results = []

    for path in resolve_image_paths(path):
        results.append(predict(path))

    output_dir.mkdir(exist_ok=True)
    save_to_file(output_dir / output_file, results)
    print("Daily job finished!")


