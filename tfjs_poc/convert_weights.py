"""Convert .npz weights to TF.js format (model.json + binary shard).

Usage:
    python tfjs_poc/convert_weights.py
"""

import json
import struct
from pathlib import Path

import numpy as np


def main():
    weights_dir = Path("tfjs_poc/weights")
    npz_path = weights_dir / "model.npz"

    data = np.load(npz_path)
    keys = sorted(data.files)

    # Build weight entries and concatenated binary buffer
    weight_specs = []
    buffer = bytearray()

    for key in keys:
        arr = data[key].astype(np.float32)
        raw = arr.tobytes()
        weight_specs.append(
            {
                "name": key,
                "shape": list(arr.shape),
                "dtype": "float32",
            }
        )
        buffer.extend(raw)

    total_bytes = len(buffer)
    shard_name = "group1-shard1of1.bin"

    # TF.js model.json format
    manifest = {
        "format": "layers-model",
        "generatedBy": "jaxrl-tfjs-poc",
        "convertedBy": "manual",
        "weightsManifest": [
            {
                "paths": [shard_name],
                "weights": weight_specs,
            }
        ],
    }

    with open(weights_dir / "model.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(weights_dir / shard_name, "wb") as f:
        f.write(buffer)

    print(f"Wrote {len(weight_specs)} weights ({total_bytes / 1024 / 1024:.1f} MB)")
    print(f"  {weights_dir / 'model.json'}")
    print(f"  {weights_dir / shard_name}")


if __name__ == "__main__":
    main()
