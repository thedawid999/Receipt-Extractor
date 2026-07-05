from .utils import save_to_file, resolve_image_paths, load_config
from .scheduler import start_scheduler, daily_job

config = load_config()

choice = input(
"""
1 - Process once
2 - Start scheduler
Choice:
"""
)

if choice == "1":
    daily_job(config)
elif choice == "2":
    start_scheduler(config)


