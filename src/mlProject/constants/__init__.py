

from pathlib import Path

CONFIG_FILE_PATH = Path(r"D:\DataScience\Ml flow project\ml-project-withmlflow\config\config.yaml")
PARAMS_FILE_PATH = Path(r"D:\DataScience\Ml flow project\ml-project-withmlflow\params.yaml")
SCHEMA_FILE_PATH = Path(r"D:\DataScience\Ml flow project\ml-project-withmlflow\schema.yaml")


assert CONFIG_FILE_PATH.exists(), f"Config file not found at {CONFIG_FILE_PATH}"
assert PARAMS_FILE_PATH.exists(), f"Params file not found at {PARAMS_FILE_PATH}"
assert SCHEMA_FILE_PATH.exists(), f"Schema file not found at {SCHEMA_FILE_PATH}"

