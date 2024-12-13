import json
import os
import sys

PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, "..", ".."))
sys.path.append(PROJECT_SRC_PATH)

from features.street import download  # noqa: E402
from util import download_all_nuts  # noqa: E402

# function parameters are passed by slurm-pipeline via stdin
params = json.load(sys.stdin)
print(params)

download_all_nuts(download, **params)
