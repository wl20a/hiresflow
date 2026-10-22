# Contributing

Issues, model-compatibility reports, benchmark reproductions, and pull requests
are welcome.

## Development checks

Build the core without CUDA or ONNX Runtime:

```bash
HIRESFLOW_SKIP_ORT=1 bash scripts/fetch_deps.sh
cmake -S . -B build-noort \
  -DHIRESFLOW_ENABLE_ORT=OFF \
  -DHIRESFLOW_BUILD_TESTS=ON
cmake --build build-noort -j
ctest --test-dir build-noort --output-on-failure
```

Build with the pinned ONNX Runtime dependency:

```bash
bash scripts/fetch_deps.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Check scripts and Python syntax:

```bash
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py python/hiresflow/*.py examples/*.py
```

## Pull requests

- Keep changes scoped and explain the problem being solved.
- Add or update tests for behavior changes.
- Preserve source-image coordinate correctness.
- State performance claims with hardware, model, command, and raw reports.
- Do not include model weights, downloaded dependencies, private images, or
  generated benchmark outputs.
