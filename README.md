# Telemetry Viewer

Interactive Streamlit app for inspecting telemetry.

## Run

```bash
uv run streamlit run telemetry_viewer.py
```

## Features

- Select site and unit CSV.
- Optionally start directly from a CSV upload.
- Plot one or multiple numeric sensors.
- Standardize raw column variants to canonical names using `COMBINE_PAIRS`.
- Overlay non-`OK` error events on a dedicated error axis.
- Compute rolling telemetry using minute/hour windows (`mean`, `median`, `min`, `max`).
- Downsample modes:
  - `skip`: keep every Nth row.
  - `time-bucket`: aggregate values by bucket (`mean`/`median`).
- Optional min/max envelope overlay in `time-bucket` mode.
- Optional z-score normalization so sensors with different scales are easier to compare.
- Show error counts and map codes to descriptions from `ERRORS.md` (if present).
- Persist sidebar config in `.config.json` so values are restored after refresh/restart.
- Uses Plotly WebGL traces and adaptive decimation (`Max points per trace`) for smoother interaction on large series.

## CSV Validation

- A custom CSV is accepted when it has `datetime` or `timestamp`.
- Any numeric columns are accepted as sensor candidates.
