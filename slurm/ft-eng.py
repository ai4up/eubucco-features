import json
import os
import sys

PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, "..", ".."))
sys.path.append(PROJECT_SRC_PATH)

from features.pipeline import execute_feature_pipeline  # noqa: E402

# function parameters are passed by slurm-pipeline via stdin
params = json.load(sys.stdin)
print(params)

execute_feature_pipeline(**params)
