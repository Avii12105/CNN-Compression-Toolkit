"""Fresh-process validation for exported ONNX artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_CHILD_SCRIPT = """
import json
import sys
import numpy as np
import onnxruntime as ort

model_path, input_name, output_name, shape_text = sys.argv[1:]
shape = tuple(int(value) for value in shape_text.split(','))
if 'CPUExecutionProvider' not in ort.get_available_providers():
    raise RuntimeError('CPUExecutionProvider is unavailable')
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
outputs = session.run([output_name], {input_name: np.zeros(shape, dtype=np.float32)})
output = np.asarray(outputs[0])
expected_shape = (shape[0], 10)
if tuple(output.shape) != expected_shape:
    raise RuntimeError(f'Unexpected output shape: {tuple(output.shape)}')
if not np.isfinite(output).all():
    raise RuntimeError('Output contains non-finite values')
print(json.dumps({'passed': True, 'output_shape': list(output.shape), 'finite_output': True}))
"""


def validate_fresh_process(
    model_path: str | Path,
    input_name: str = "input",
    output_name: str = "logits",
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
) -> dict[str, Any]:
    """Reload an ONNX artifact in a separate Python process and validate inference."""

    command = [
        sys.executable,
        "-c",
        _CHILD_SCRIPT,
        str(model_path),
        input_name,
        output_name,
        ",".join(str(value) for value in input_shape),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result: dict[str, Any] = {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0:
        try:
            result.update(json.loads(completed.stdout))
        except json.JSONDecodeError:
            result.update({"passed": False, "reason": "child output was not valid JSON"})
    else:
        result["reason"] = "fresh process exited with an error"
    return result