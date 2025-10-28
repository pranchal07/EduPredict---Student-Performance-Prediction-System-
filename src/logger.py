import logging
from datetime import datetime
from pathlib import Path

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = Path.cwd() / "logs"
logs_path.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = logs_path / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")
