from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from constants import COMBINE_PAIRS, DAIKIN_OUTDOOR_COMMON_COLS

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
UPLOAD_CACHE_DIR = TOOLS_DIR / ".cache"
DEFAULT_DATA_DIR = ROOT / "data" / "outdoor"
DEFAULT_ERROR_DOC = TOOLS_DIR / "ERRORS.md"
CONFIG_PATH = TOOLS_DIR / ".config.json"
EXCLUDED_SENSOR_COLUMNS = {
    "Unit_Id",
    "timestamp",
    "datetime",
    "Outdoor Type",
    "Error Code",
    "System ID",
    "is_filled",
    "time_gap_minutes",
    "time_gap_category",
}
ROLLING_METHODS = ["mean", "median", "min", "max"]
WINDOW_UNITS = ["minutes", "hours"]
DAIKIN_SCHEMA_MATCH_THRESHOLD = 8


def load_persisted_config() -> dict[str, object]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_persisted_config(config: dict[str, object]) -> None:
    try:
        CONFIG_PATH.write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
    except OSError:
        pass


def valid_option(value: object, options: list[str], fallback: str) -> str:
    if isinstance(value, str) and value in options:
        return value
    return fallback


def valid_int(
    value: object,
    minimum: int,
    maximum: int,
    fallback: int,
) -> int:
    try:
        numeric = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return min(max(numeric, minimum), maximum)


def valid_multiselect(
    value: object, options: list[str], fallback: list[str]
) -> list[str]:
    if not isinstance(value, list):
        return fallback
    selected = [v for v in value if isinstance(v, str) and v in options]
    return selected if selected else fallback


def is_numeric_dtype_name(dtype_name: str) -> bool:
    prefixes = ("Int", "UInt", "Float", "Decimal")
    return dtype_name.startswith(prefixes)


def persist_uploaded_csv(uploaded_file: object) -> str:
    if uploaded_file is None:
        return ""
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.md5(file_bytes).hexdigest()[:12]
    original_name = Path(uploaded_file.name).name
    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_CACHE_DIR / f"{digest}_{original_name}"
    if not target.exists():
        target.write_bytes(file_bytes)
    return str(target)


def validate_csv_schema(schema_map: dict[str, str]) -> tuple[bool, str]:
    cols = set(schema_map.keys())
    has_time = ("datetime" in cols) or ("timestamp" in cols)
    numeric_cols = [
        name
        for name, dtype_name in schema_map.items()
        if is_numeric_dtype_name(dtype_name)
    ]
    if not has_time:
        return False, "CSV must contain `datetime` or `timestamp` column."
    if not numeric_cols:
        return False, "CSV must contain at least one numeric column."
    return True, ""


@dataclass(frozen=True)
class RollingConfig:
    enabled: bool
    method: str
    window_size: int
    window_unit: str

    @property
    def window_expr(self) -> str:
        suffix = "m" if self.window_unit == "minutes" else "h"
        return f"{self.window_size}{suffix}"


def parse_error_markdown(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^#\s+([A-Z0-9]+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    descriptions: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end]
        bullets = re.findall(r"^\*\s+(.+)$", chunk, flags=re.MULTILINE)
        descriptions[match.group(1)] = " | ".join(bullets) if bullets else ""
    return descriptions


@st.cache_data(show_spinner=False)
def list_unit_files(data_dir: str) -> dict[str, list[Path]]:
    root = Path(data_dir)
    files = sorted(root.glob("*/*.csv"))
    by_site: dict[str, list[Path]] = {}
    for file in files:
        by_site.setdefault(file.parent.name, []).append(file)
    return by_site


@st.cache_data(show_spinner=False)
def raw_schema_map(csv_path: str) -> dict[str, str]:
    schema = pl.scan_csv(csv_path, try_parse_dates=False).collect_schema()
    return {name: str(dtype) for name, dtype in schema.items()}


def standardize_schema_map(schema_map: dict[str, str]) -> dict[str, str]:
    out = dict(schema_map)

    if "INV1 comp. body temp (°C)" in out and "INV2 comp. body temp (°C)" in out:
        out["INV comp. body temp (°C)"] = out["INV1 comp. body temp (°C)"]

    for col0, col1, new_col in COMBINE_PAIRS:
        if col0 in out:
            out[new_col] = out.pop(col0)
        elif col1 in out:
            out[new_col] = out.pop(col1)

    keep_cols = [col for col in DAIKIN_OUTDOOR_COMMON_COLS if col in out]
    if len(keep_cols) >= DAIKIN_SCHEMA_MATCH_THRESHOLD:
        return {col: out[col] for col in keep_cols}
    return out


def resolve_required_raw_columns(
    raw_cols: set[str], required_standard_cols: list[str]
) -> list[str]:
    required: set[str] = set()
    if "datetime" in raw_cols:
        required.add("datetime")
    elif "timestamp" in raw_cols:
        required.add("timestamp")

    for col in required_standard_cols:
        if col in raw_cols:
            required.add(col)
            if col == "Error Code":
                # Canonical column already present; avoid adding alias columns
                # that would later rename into a duplicate "Error Code".
                continue
        for col0, col1, new_col in COMBINE_PAIRS:
            if new_col != col:
                continue
            if col0 in raw_cols:
                required.add(col0)
            elif col1 in raw_cols:
                required.add(col1)
        if col == "Compressor Surface Temp. (°C)":
            if (
                "INV1 comp. body temp (°C)" in raw_cols
                and "INV2 comp. body temp (°C)" in raw_cols
            ):
                required.add("INV1 comp. body temp (°C)")
                required.add("INV2 comp. body temp (°C)")

    if "Error Code" in raw_cols:
        required.add("Error Code")
    return sorted(required)


def standardize_lazy_frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    out = lf
    cols = set(out.collect_schema().names())

    if "INV1 comp. body temp (°C)" in cols and "INV2 comp. body temp (°C)" in cols:
        inv1 = pl.col("INV1 comp. body temp (°C)")
        inv2 = pl.col("INV2 comp. body temp (°C)")
        out = out.with_columns(
            pl.when(inv1.is_null() & inv2.is_null())
            .then(None)
            .when((inv1 == 0) & (inv2 == 0))
            .then(0.0)
            .when((inv1.is_null() | (inv1 == 0)) & inv2.is_not_null() & (inv2 != 0))
            .then(inv2)
            .when((inv2.is_null() | (inv2 == 0)) & inv1.is_not_null() & (inv1 != 0))
            .then(inv1)
            .otherwise((inv1 + inv2) / 2.0)
            .alias("INV comp. body temp (°C)")
        )

    rename_map: dict[str, str] = {}
    cols = set(out.collect_schema().names())
    for col0, col1, new_col in COMBINE_PAIRS:
        if new_col in cols:
            # Prefer existing canonical column; do not create duplicates.
            continue
        if col0 in cols:
            rename_map[col0] = new_col
        elif col1 in cols:
            rename_map[col1] = new_col
    if rename_map:
        out = out.rename(rename_map)

    keep_cols = [
        col for col in DAIKIN_OUTDOOR_COMMON_COLS if col in out.collect_schema().names()
    ]
    if len(keep_cols) >= DAIKIN_SCHEMA_MATCH_THRESHOLD:
        return out.select(keep_cols)
    return out


@st.cache_data(show_spinner=False)
def load_unit_frame(csv_path: str, selected_sensors: tuple[str, ...]) -> pl.DataFrame:
    raw_cols = set(raw_schema_map(csv_path).keys())
    required_standard_cols = ["Error Code", "datetime", "timestamp", *selected_sensors]
    required_raw_cols = resolve_required_raw_columns(raw_cols, required_standard_cols)

    lf = pl.scan_csv(csv_path, try_parse_dates=False).select(
        [pl.col(c) for c in required_raw_cols if c in raw_cols]
    )
    lf = standardize_lazy_frame(lf)
    cols = set(lf.collect_schema().names())
    if "datetime" in cols:
        lf = lf.with_columns(
            pl.col("datetime")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("_dt")
        )
    elif "timestamp" in cols:
        lf = lf.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("_dt")
        )
    else:
        raise ValueError("Expected either 'datetime' or 'timestamp' column.")

    if "Error Code" not in cols:
        lf = lf.with_columns(pl.lit("OK").alias("Error Code"))

    required_out = ["_dt", "Error Code", *selected_sensors]
    available_out = [c for c in required_out if c in lf.collect_schema().names()]
    return lf.select(available_out).sort("_dt").collect()


def numeric_sensor_columns(df: pl.DataFrame) -> list[str]:
    cols: list[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in EXCLUDED_SENSOR_COLUMNS or name.startswith("_"):
            continue
        if dtype.is_numeric():
            cols.append(name)
    return cols


def decimate_rows(
    df: pl.DataFrame, max_points_per_trace: int
) -> tuple[pl.DataFrame, int]:
    if max_points_per_trace < 1:
        return df, 1
    if df.height <= max_points_per_trace:
        return df, 1
    step = ceil(df.height / max_points_per_trace)
    out = (
        df.with_row_index("__row_idx")
        .filter((pl.col("__row_idx") % step) == 0)
        .drop("__row_idx")
    )
    return out, step


def apply_rolling(
    df: pl.DataFrame, sensors: list[str], cfg: RollingConfig
) -> tuple[pl.DataFrame, list[str]]:
    if not cfg.enabled:
        return df, sensors

    exprs = []
    rolled_cols: list[str] = []
    for sensor in sensors:
        col_name = f"{sensor} [{cfg.method} {cfg.window_expr}]"
        if cfg.method == "mean":
            expr = pl.col(sensor).rolling_mean_by("_dt", window_size=cfg.window_expr)
        elif cfg.method == "median":
            expr = pl.col(sensor).rolling_median_by("_dt", window_size=cfg.window_expr)
        elif cfg.method == "min":
            expr = pl.col(sensor).rolling_min_by("_dt", window_size=cfg.window_expr)
        else:
            expr = pl.col(sensor).rolling_max_by("_dt", window_size=cfg.window_expr)
        exprs.append(expr.alias(col_name))
        rolled_cols.append(col_name)
    return df.with_columns(exprs), rolled_cols


def normalize_columns(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    exprs = []
    for col in cols:
        mu = pl.col(col).mean()
        sigma = pl.col(col).std()
        exprs.append(
            pl.when(sigma > 0)
            .then((pl.col(col) - mu) / sigma)
            .otherwise(None)
            .fill_nan(None)
            .alias(col)
        )
    return df.with_columns(exprs)


def error_event_frame(df: pl.DataFrame, code_col: str = "Error Code") -> pl.DataFrame:
    if code_col not in df.columns:
        return pl.DataFrame(
            {"_dt": [], "Error Code": []},
            schema={"_dt": pl.Datetime, "Error Code": pl.String},
        )
    return df.filter(
        pl.col(code_col).is_not_null()
        & (pl.col(code_col) != "OK")
        & (pl.col(code_col) != "")
    ).select(["_dt", code_col])


def build_plot(
    df: pl.DataFrame,
    sensors: list[str],
    errors: pl.DataFrame,
    show_error_overlay: bool,
    normalize: bool,
) -> go.Figure:
    plot_df = df.select(["_dt", "Error Code", *sensors])
    if normalize:
        plot_df = normalize_columns(plot_df, sensors)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.04,
    )
    x = plot_df.get_column("_dt").to_list()
    for sensor in sensors:
        y = plot_df.get_column(sensor).to_numpy()
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines",
                name=sensor,
                line={"width": 1.2},
            ),
            row=1,
            col=1,
        )

    if show_error_overlay and errors.height > 0:
        err_times = errors.get_column("_dt").to_list()
        err_codes = errors.get_column("Error Code").to_list()
        fig.add_trace(
            go.Scattergl(
                x=err_times,
                y=err_codes,
                mode="markers",
                name="Error Code",
                marker={"size": 7},
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Sensor Telemetry",
        height=760,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )
    fig.update_yaxes(
        title_text="z-score" if normalize else "sensor value", row=1, col=1
    )
    fig.update_yaxes(title_text="Error", type="category", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    return fig


def main() -> None:
    st.set_page_config(page_title="Telemetry Viewer", layout="wide")
    st.title("Sensor Telemetry Viewer")
    st.caption("Visualize telemetry, overlay error codes, and compute rolling windows.")

    saved = load_persisted_config()
    if "_cfg_inited" not in st.session_state:
        st.session_state["_cfg_inited"] = True
        st.session_state["cfg_data_dir"] = str(
            saved.get("data_dir", str(DEFAULT_DATA_DIR))
        )
        st.session_state["cfg_error_doc"] = str(
            saved.get("error_doc", str(DEFAULT_ERROR_DOC))
        )
        st.session_state["cfg_csv_path"] = str(saved.get("csv_path", ""))
        st.session_state["cfg_site"] = saved.get("site")
        st.session_state["cfg_unit"] = saved.get("unit")
        st.session_state["cfg_selected_sensors"] = saved.get("selected_sensors", [])
        st.session_state["cfg_rolling_enabled"] = bool(
            saved.get("rolling_enabled", False)
        )
        st.session_state["cfg_rolling_method"] = str(
            saved.get("rolling_method", ROLLING_METHODS[0])
        )
        st.session_state["cfg_window_size"] = valid_int(
            saved.get("window_size"), 1, 24 * 60, 60
        )
        st.session_state["cfg_window_unit"] = str(
            saved.get("window_unit", WINDOW_UNITS[0])
        )
        st.session_state["cfg_show_errors"] = bool(saved.get("show_errors", True))
        st.session_state["cfg_normalize"] = bool(saved.get("normalize", False))
        st.session_state["cfg_frame_step"] = valid_int(
            saved.get("frame_step"), 1, 30, 1
        )
        st.session_state["cfg_max_points_per_trace"] = valid_int(
            saved.get("max_points_per_trace"), 500, 200_000, 6_000
        )

    data_dir = st.sidebar.text_input("Data directory", key="cfg_data_dir")
    error_doc = st.sidebar.text_input("Error doc path", key="cfg_error_doc")
    csv_path_input = st.sidebar.text_input("CSV path (optional)", key="cfg_csv_path")
    uploaded_csv = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    uploaded_csv_path = persist_uploaded_csv(uploaded_csv)

    source_label = ""
    if uploaded_csv_path:
        unit_path = uploaded_csv_path
        source_label = f"Uploaded: {Path(uploaded_csv_path).name}"
    elif csv_path_input.strip():
        candidate = Path(csv_path_input).expanduser()
        if not candidate.exists() or not candidate.is_file():
            st.error(f"CSV path does not exist or is not a file: `{candidate}`")
            return
        if candidate.suffix.lower() != ".csv":
            st.error(f"CSV path is not a .csv file: `{candidate}`")
            return
        unit_path = str(candidate.resolve())
        source_label = f"Path: {candidate.name}"
    else:
        site_map = list_unit_files(data_dir)
        if not site_map:
            st.error(
                f"No CSV files found in {data_dir}. Expected layout: data/outdoor/<site>/<unit>.csv"
            )
            return

        sites = sorted(site_map.keys())
        st.session_state["cfg_site"] = valid_option(
            st.session_state.get("cfg_site"), sites, sites[0]
        )
        site = st.sidebar.selectbox("Site", options=sites, key="cfg_site")
        unit_paths = site_map[site]
        unit_labels = [p.stem for p in unit_paths]
        st.session_state["cfg_unit"] = valid_option(
            st.session_state.get("cfg_unit"), unit_labels, unit_labels[0]
        )
        selected_unit = st.sidebar.selectbox(
            "Unit", options=unit_labels, key="cfg_unit"
        )
        unit_path = str(unit_paths[unit_labels.index(selected_unit)])
        source_label = f"{site}/{selected_unit}"

    raw_schema = raw_schema_map(unit_path)
    is_valid, validation_msg = validate_csv_schema(raw_schema)
    if not is_valid:
        st.error(validation_msg)
        return

    standardized_schema = standardize_schema_map(raw_schema)
    sensors = [
        name
        for name, dtype_name in standardized_schema.items()
        if name not in EXCLUDED_SENSOR_COLUMNS
        and not name.startswith("_")
        and is_numeric_dtype_name(dtype_name)
    ]
    default_sensors = sensors[:2] if len(sensors) >= 2 else sensors
    st.session_state["cfg_selected_sensors"] = valid_multiselect(
        st.session_state.get("cfg_selected_sensors"), sensors, default_sensors
    )
    selected_sensors = st.sidebar.multiselect(
        "Sensors", options=sensors, key="cfg_selected_sensors"
    )

    st.session_state["cfg_rolling_method"] = valid_option(
        st.session_state.get("cfg_rolling_method"), ROLLING_METHODS, ROLLING_METHODS[0]
    )
    st.session_state["cfg_window_size"] = valid_int(
        st.session_state.get("cfg_window_size"), 1, 24 * 60, 60
    )
    st.session_state["cfg_window_unit"] = valid_option(
        st.session_state.get("cfg_window_unit"), WINDOW_UNITS, WINDOW_UNITS[0]
    )
    st.session_state["cfg_frame_step"] = valid_int(
        st.session_state.get("cfg_frame_step"), 1, 30, 1
    )
    st.session_state["cfg_max_points_per_trace"] = valid_int(
        st.session_state.get("cfg_max_points_per_trace"), 500, 200_000, 6_000
    )

    rolling_enabled = st.sidebar.checkbox(
        "Rolling telemetry", key="cfg_rolling_enabled"
    )
    rolling_method = st.sidebar.selectbox(
        "Rolling method",
        options=ROLLING_METHODS,
        key="cfg_rolling_method",
    )
    window_size = st.sidebar.number_input(
        "Window size",
        min_value=1,
        max_value=24 * 60,
        step=1,
        key="cfg_window_size",
    )
    window_unit = st.sidebar.radio(
        "Window unit",
        options=WINDOW_UNITS,
        horizontal=True,
        key="cfg_window_unit",
    )
    show_errors = st.sidebar.checkbox("Show error code overlay", key="cfg_show_errors")
    normalize = st.sidebar.checkbox("Normalize sensors (z-score)", key="cfg_normalize")
    frame_step = st.sidebar.slider(
        "Downsample every N rows",
        min_value=1,
        max_value=30,
        step=1,
        key="cfg_frame_step",
    )
    max_points_per_trace = st.sidebar.number_input(
        "Max points per trace",
        min_value=500,
        max_value=200_000,
        step=500,
        key="cfg_max_points_per_trace",
    )

    rolling_cfg = RollingConfig(
        enabled=rolling_enabled,
        method=rolling_method,
        window_size=int(window_size),
        window_unit=window_unit,
    )

    save_persisted_config(
        {
            "data_dir": data_dir,
            "error_doc": error_doc,
            "csv_path": csv_path_input,
            "site": st.session_state.get("cfg_site"),
            "unit": st.session_state.get("cfg_unit"),
            "selected_sensors": selected_sensors,
            "rolling_enabled": rolling_cfg.enabled,
            "rolling_method": rolling_cfg.method,
            "window_size": int(rolling_cfg.window_size),
            "window_unit": rolling_cfg.window_unit,
            "show_errors": show_errors,
            "normalize": normalize,
            "frame_step": int(frame_step),
            "max_points_per_trace": int(max_points_per_trace),
        }
    )

    if not selected_sensors:
        st.info("Pick at least one sensor.")
        return

    df = load_unit_frame(unit_path, tuple(selected_sensors))
    min_dt = df.select(pl.col("_dt").min()).item()
    max_dt = df.select(pl.col("_dt").max()).item()
    st.sidebar.write(f"Range: `{min_dt}` to `{max_dt}`")

    if frame_step > 1:
        filtered = (
            df.with_row_index("__row_idx")
            .filter((pl.col("__row_idx") % frame_step) == 0)
            .drop("__row_idx")
        )
    else:
        filtered = df

    errors = error_event_frame(filtered)
    plotted_df, plotted_cols = apply_rolling(filtered, selected_sensors, rolling_cfg)
    plotted_df, auto_step = decimate_rows(plotted_df, int(max_points_per_trace))
    if auto_step > 1:
        st.caption(
            f"Adaptive decimation applied: keeping ~1/{auto_step} rows "
            f"({plotted_df.height:,} plotted points per trace max)."
        )
    if not show_errors:
        st.caption("Error overlay is disabled. Enable `Show error code overlay` to display error markers.")
    elif errors.height == 0:
        st.caption("No non-OK error events in selected range.")
    fig = build_plot(plotted_df, plotted_cols, errors, show_errors, normalize)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Selected Unit")
        st.write(f"Source: `{source_label}`")
        st.write(f"`{unit_path}`")
        st.write(f"Rows: `{filtered.height:,}`")
        st.dataframe(
            filtered.select(["_dt", "Error Code", *selected_sensors]).head(100),
            width="stretch",
        )
    with c2:
        st.subheader("Error Summary")
        if errors.height == 0:
            st.write("No non-OK errors in selected unit/range.")
        else:
            counts = (
                errors.group_by("Error Code")
                .len()
                .sort("len", descending=True)
                .rename({"len": "count"})
            )
            st.dataframe(counts, width="stretch")

            descriptions = parse_error_markdown(Path(error_doc))
            if descriptions:
                codes = counts.get_column("Error Code").to_list()
                desc_frame = pl.DataFrame(
                    {
                        "Error Code": codes,
                        "Description": [descriptions.get(code, "") for code in codes],
                    }
                )
                st.subheader("Error Guide")
                st.dataframe(desc_frame, width="stretch")


if __name__ == "__main__":
    main()
