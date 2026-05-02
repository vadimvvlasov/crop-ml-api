"""Build PredictRequest JSON from RS HLS NPZ + parcels GPKG (real features + labels).

Maps ``crop_class`` strings from GPKG to eval ``class_code`` using
``classmapping_eval.csv``. ``Pasture`` and ``Forest Plantation`` fold into eval
``Other`` (code 2).

Loads crop labels for all polygons via SQLite; decodes only selected geometries
(GeoPackage binary envelope + ISO WKB) via Shapely.
Reflectance gaps (NaN) are filled like ``HLSStitchedDataset`` (linear along time,
ffill/bfill) before JSON export; payload uses strict JSON (no NaN literals).

Usage::

    uv sync --group dev
    uv run python scripts/export_rs_hls_fixtures.py

Environment ``RESEARCH_CROPS_ROOT`` or CLI ``--research-root`` selects checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from shapely import wkb

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.hls_preprocess import interpolate_features_ntc  # noqa: E402

RS_REL = (
    "dags/data/hls/interim/"
    "hls_output_2024_Rio-Grande-do-Sul_32722_avg_241020_brazil-241001-250331"
)


def resolve_research_root(cli: Path | None) -> Path:
    if cli is not None:
        return cli.expanduser().resolve()
    env = os.environ.get("RESEARCH_CROPS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        Path("/home/vadim/work/agro-mon/repos/research-crops"),
        Path(__file__).resolve().parents[3] / "research-crops",
    ]
    for c in candidates:
        if (c / "dags/data/hls").is_dir():
            return c
    raise FileNotFoundError(
        "Set RESEARCH_CROPS_ROOT or pass --research-root pointing at research-crops."
    )


def load_eval_name_to_code(classmapping_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    name_to_code: dict[str, int] = {}
    code_to_name: dict[int, str] = {}
    with classmapping_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = int(row["class_code"])
            name = row["class_name"].strip()
            name_to_code[name.lower()] = code
            code_to_name[code] = name
    return name_to_code, code_to_name


def crop_class_to_eval_id(raw: str, name_to_code: dict[str, int]) -> int:
    key = raw.strip().lower()
    alias = {"soybeans": "soybean", "maize": "corn"}
    key = alias.get(key, key)
    if key in name_to_code:
        return name_to_code[key]
    if key in frozenset({"pasture", "forest plantation"}):
        return name_to_code["other"]
    raise ValueError(f"Unmapped crop_class: {raw!r}")


def gpkg_feature_table(con: sqlite3.Connection) -> str:
    row = con.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type='features' LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError("No features table in GPKG")
    return row[0]


def parse_gpkg_geom(blob: bytes):
    """Strip GeoPackage binary header; parse remainder as ISO WKB."""
    if len(blob) < 8:
        raise ValueError("Geometry blob too short")
    if blob[:2] != b"GP":
        return wkb.loads(blob)
    flags = blob[3]
    env_ind = (flags >> 1) & 7
    env_sizes = {0: 0, 1: 32, 2: 48, 3: 64, 4: 48}
    skip = 8 + env_sizes.get(env_ind, 0)
    return wkb.loads(blob[skip:])


def load_crop_class_by_field(gpkg_path: Path) -> dict[str, str]:
    """field_id -> crop_class (full layer scan, no geometry decode)."""
    con = sqlite3.connect(gpkg_path)
    try:
        tbl = gpkg_feature_table(con)
        cur = con.execute(f'SELECT field_id, crop_class FROM "{tbl}"')
        return {str(fid): crop_class for fid, crop_class in cur}
    finally:
        con.close()


def load_centroids_by_field(gpkg_path: Path, field_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Centroids as (lat, lon) WGS84 for requested ``field_id`` values."""
    if not field_ids:
        return {}
    con = sqlite3.connect(gpkg_path)
    try:
        tbl = gpkg_feature_table(con)
        placeholders = ",".join("?" * len(field_ids))
        sql = f'SELECT field_id, geom FROM "{tbl}" WHERE field_id IN ({placeholders})'
        cur = con.execute(sql, field_ids)
        out: dict[str, tuple[float, float]] = {}
        for fid, geom_blob in cur:
            geom = parse_gpkg_geom(bytes(geom_blob))
            c = geom.centroid
            out[str(fid)] = (float(c.y), float(c.x))
        missing = set(field_ids) - set(out.keys())
        if missing:
            raise KeyError(f"GPKG missing field_id rows: {list(missing)[:5]}...")
        return out
    finally:
        con.close()


def weeks_to_iso_week_numbers(week_raw: np.ndarray) -> list[int]:
    starts = [str(w).split("-")[0] for w in week_raw]
    out: list[int] = []
    for s in starts:
        dt = datetime.strptime(s, "%y%m%d")
        out.append(int(dt.isocalendar()[1]))
    return out


def pick_field_indices(
    fids: np.ndarray,
    crop_by_field: dict[str, str],
    name_to_code: dict[str, int],
    limit: int,
) -> list[int]:
    """Prefer covering distinct eval ids, then distinct crop_class; fill linearly."""
    n = int(fids.shape[0])
    first_by_eval: dict[int, int] = {}
    first_by_crop: dict[str, int] = {}

    for i in range(n):
        fid_s = str(fids[i])
        crop = crop_by_field[fid_s]
        ev = crop_class_to_eval_id(crop, name_to_code)
        first_by_eval.setdefault(ev, i)
        first_by_crop.setdefault(crop, i)

    chosen: list[int] = []
    for ev in sorted(first_by_eval.keys()):
        ix = first_by_eval[ev]
        if ix not in chosen:
            chosen.append(ix)
        if len(chosen) >= limit:
            return chosen[:limit]

    for crop in sorted(first_by_crop.keys()):
        ix = first_by_crop[crop]
        if ix not in chosen:
            chosen.append(ix)
        if len(chosen) >= limit:
            return chosen[:limit]

    for i in range(n):
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-root", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures",
    )
    p.add_argument("--limit", type=int, default=10)
    return p.parse_args()


def write_fixtures(
    out_dir: Path,
    npz_path: Path,
    gpkg_path: Path,
    mapping_path: Path,
    feats: np.ndarray,
    fids: np.ndarray,
    week_of_year: list[int],
    crop_by_field: dict[str, str],
    centroid_by_field: dict[str, tuple[float, float]],
    name_to_code: dict[str, int],
    code_to_name: dict[int, str],
    idxs: list[int],
) -> None:
    sel_tnc = np.asarray(feats[:, idxs, :], dtype=np.float32)
    batch_ntc = np.transpose(sel_tnc, (1, 0, 2))
    filled_ntc = interpolate_features_ntc(batch_ntc)
    features_out = filled_ntc.tolist()

    locations: list[list[float]] = []
    meta_rows: list[dict] = []

    for j, i in enumerate(idxs):
        fid_s = str(fids[i])
        crop = crop_by_field[fid_s]
        ev = crop_class_to_eval_id(crop, name_to_code)
        lat, lon = centroid_by_field[fid_s]
        locations.append([lat, lon])
        meta_rows.append(
            {
                "batch_index": j,
                "npz_axis1_index": i,
                "field_id": fid_s,
                "crop_class": crop,
                "eval_class_id": ev,
                "eval_class_name": code_to_name[ev],
            }
        )

    payload = {
        "features": features_out,
        "week_of_year": week_of_year,
        "location": locations,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "rs_hls_predict_request.json"
    meta_path = out_dir / "rs_hls_fixture_meta.json"
    pred_path.write_text(
        json.dumps(payload, allow_nan=False),
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(
            {
                "source_npz": str(npz_path),
                "source_gpkg": str(gpkg_path),
                "classmapping_eval": str(mapping_path),
                "n_exported": len(idxs),
                "week_note": "ISO week from NPZ week strings (YYMMDD-YYMMDD), first token",
                "preprocess_note": "NaNs interpolated like research-crops HLSStitchedDataset (before API normalization)",
                "samples": meta_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {pred_path}")
    print(f"Wrote {meta_path}")


def main() -> None:
    args = parse_args()
    root = resolve_research_root(args.research_root)
    interim = root / RS_REL
    npz_path = interim / (
        "features_combined_output_2024_Rio-Grande-do-Sul_32722_avg_241020_brazil-241001-250331.npz"
    )
    gpkg_path = interim / "output_2024_Rio-Grande-do-Sul_32722_avg_241020_brazil.gpkg"
    mapping_path = root / "dags/data/hls/external/classmapping_eval.csv"

    name_to_code, code_to_name = load_eval_name_to_code(mapping_path)
    crop_by_field = load_crop_class_by_field(gpkg_path)

    z = np.load(npz_path, mmap_mode="r", allow_pickle=True)
    feats = z["features"]
    fids = np.asarray(z["field_id"]).ravel()
    week_raw = np.asarray(z["week"]).ravel()
    week_of_year = weeks_to_iso_week_numbers(week_raw)

    idxs = pick_field_indices(fids, crop_by_field, name_to_code, args.limit)
    selected_fids = [str(fids[i]) for i in idxs]
    centroid_by_field = load_centroids_by_field(gpkg_path, selected_fids)

    write_fixtures(
        args.out_dir,
        npz_path,
        gpkg_path,
        mapping_path,
        feats,
        fids,
        week_of_year,
        crop_by_field,
        centroid_by_field,
        name_to_code,
        code_to_name,
        idxs,
    )


if __name__ == "__main__":
    main()
