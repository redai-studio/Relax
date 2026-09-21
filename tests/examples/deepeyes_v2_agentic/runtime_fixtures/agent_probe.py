# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import json
import os
from pathlib import Path

from app.search_config import SEARCH_CONFIG_ENV, load_search_config


parser = argparse.ArgumentParser()
parser.add_argument("--input-json", type=Path, required=True)
parser.add_argument("--output-json", type=Path, required=True)
args = parser.parse_args()
json.loads(args.input_json.read_text(encoding="utf-8"))
config = load_search_config()
metadata = {
    "backend": config.backend,
    "config_path": os.environ[SEARCH_CONFIG_ENV],
    "auth": os.environ.get(config.auth.env) if config.backend == "external" and config.auth else None,
}
args.output_json.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
