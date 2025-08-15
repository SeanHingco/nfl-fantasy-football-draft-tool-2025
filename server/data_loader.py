"""
data_loader.py
================

This module provides utilities for loading and cleaning fantasy football
projection and average draft position (ADP) data.  It consolidates multiple
projection files, removes empty rows, extracts the relevant position and
ADP fields, and returns a clean pandas ``DataFrame`` suitable for
consumption by downstream logic (e.g., Monte Carlo draft simulations or a
FastAPI backend).

The main entry point is :func:`load_clean_player_data`.

Example
-------

```python
from data_loader import load_clean_player_data

# Path to your data directory containing the projection CSVs and ADP file
data_dir = "/path/to/my/data"

df = load_clean_player_data(data_dir)
print(df.head())
```
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, List
from draft_models import Player

import pandas as pd

def _read_projection_files(data_dir: str) -> pd.DataFrame:
    """Internal helper to read and concatenate all projection CSV files.

    Looks for files matching ``FantasyPros_Fantasy_Football_Projections_*.csv``
    within ``data_dir``.  The special ``FLX`` file is ignored to prevent
    duplication since the flex position is implicitly covered by the RB, WR
    and TE projections.

    The function adds a consistent ``Position`` column to each DataFrame
    derived from the filename suffix.

    Parameters
    ----------
    data_dir : str
        Directory containing the projection CSV files.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame of all projection files.
    """
    pattern = os.path.join(data_dir, "FantasyPros_Fantasy_Football_Projections_*.csv")
    file_paths: List[str] = glob.glob(pattern)
    frames: List[pd.DataFrame] = []

    for path in file_paths:
        fname = os.path.basename(path)
        # Skip the flex projections to avoid duplications
        if "FLX" in fname.upper():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            # Skip unreadable files gracefully
            continue
        # Derive position from filename (e.g., *_QB.csv -> QB)
        # Take everything between the final underscore and the extension
        pos = fname.rsplit("_", 1)[-1].split(".")[0].upper()
        df["Position"] = pos
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    # Concatenate all frames.  ignore_index ensures a fresh 0-based index
    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


def _clean_projections(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw projections DataFrame.

    - Strips whitespace from the ``Player`` column and drops rows
      with missing or empty player names.
    - Drops columns that are entirely NaN across all rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw projections DataFrame.

    Returns
    -------
    pandas.DataFrame
        Cleaned projections DataFrame.
    """
    if df.empty:
        return df.copy()
    # Ensure Player column is a string and strip whitespace
    df["Player"] = df["Player"].astype(str).str.strip()
    # Drop rows where Player is missing or an empty string
    df = df[df["Player"].notna() & (df["Player"] != "")].copy()
    # Remove columns that contain all NaN values
    df = df.loc[:, df.notna().any()]
    return df


def _read_and_clean_adp(data_dir: str) -> pd.DataFrame:
    """Internal helper to read and clean the ADP file.

    Expects a single file named ``FantasyPros_2025_Overall_ADP_Rankings.csv``
    in ``data_dir``.  If the file is not found or is unreadable, returns an
    empty DataFrame.

    - Drops rows with missing player names.
    - Renames the ``AVG`` column to ``ADP`` and coerces it to a numeric type.
    - Extracts the base position (e.g., ``WR`` from ``WR1``) into a
      ``Position`` column.
    - Retains only the ``Player``, ``Position`` and ``ADP`` columns.

    Parameters
    ----------
    data_dir : str
        Directory containing the ADP file.

    Returns
    -------
    pandas.DataFrame
        Cleaned ADP DataFrame.
    """
    adp_path = os.path.join(data_dir, "FantasyPros_2025_Overall_ADP_Rankings.csv")
    if not os.path.isfile(adp_path):
        return pd.DataFrame(columns=["Player", "Position", "ADP"])
    try:
        adp_df = pd.read_csv(adp_path)
    except Exception:
        return pd.DataFrame(columns=["Player", "Position", "ADP"])

    # Remove rows with missing player names
    adp_df = adp_df[adp_df["Player"].notna()].copy()
    # Rename AVG to ADP if present.  We use .get to avoid KeyError if the column
    # doesn't exist.  Similarly, rename POS to PosCategory if present.
    rename_map = {}
    if "AVG" in adp_df.columns:
        rename_map["AVG"] = "ADP"
    if "POS" in adp_df.columns:
        rename_map["POS"] = "PosCategory"
    if rename_map:
        adp_df = adp_df.rename(columns=rename_map)

    # Ensure ADP column exists; if not, create it with NaNs
    if "ADP" not in adp_df.columns:
        adp_df["ADP"] = pd.NA
    else:
        adp_df["ADP"] = pd.to_numeric(adp_df["ADP"], errors="coerce")

    # Derive Position: if PosCategory exists (e.g., WR1, RB3) extract letters;
    # otherwise fall back to POS or set as NA.  We prioritise PosCategory, then POS.
    position_series = pd.Series([pd.NA] * len(adp_df))
    if "PosCategory" in adp_df.columns:
        position_series = adp_df["PosCategory"].astype(str).str.extract(r"([A-Z]+)")[0]
    elif "POS" in adp_df.columns:
        position_series = adp_df["POS"].astype(str).str.extract(r"([A-Z]+)")[0]
    adp_df["Position"] = position_series

    # Keep only the necessary columns
    adp_df = adp_df[["Player", "Position", "ADP"]]
    return adp_df


def load_clean_player_data(data_dir: str, *, fill_missing_adp: float | None = 999) -> pd.DataFrame:
    """Load, clean and merge projection and ADP data into a single DataFrame.

    This function orchestrates reading multiple projection files and the
    single ADP file, cleaning both datasets, and merging them on the
    ``Player`` and ``Position`` fields.  It returns only the fields
    necessary for downstream simulation:

    - ``Player``: the player's name
    - ``Team``: NFL team (if present in projections)
    - ``Position``: base position (QB, RB, WR, TE, DST, K)
    - ``FPTS``: projected fantasy points from the projections
    - ``ADP``: average draft position (from the ADP file)

    Any rows lacking a ``Player``, ``Position`` or ``FPTS`` will be
    discarded.  Missing ADP values can optionally be filled with a
    placeholder (e.g., ``999``) to indicate a very late/undrafted player.

    Parameters
    ----------
    data_dir : str
        Directory containing the projection and ADP CSV files.
    fill_missing_adp : float or None, optional
        Value to fill missing ADP entries with.  If ``None``, missing ADP
        values are left as NaN.

    Returns
    -------
    pandas.DataFrame
        A cleaned DataFrame with columns ``['Player', 'Team', 'Position', 'FPTS', 'ADP']``.
    """
    # Load and clean projections
    proj_raw = _read_projection_files(data_dir)
    proj_clean = _clean_projections(proj_raw)

    # Read and clean ADP file
    adp_clean = _read_and_clean_adp(data_dir)

    # Merge on Player and Position
    if not proj_clean.empty and not adp_clean.empty:
        merged_df = proj_clean.merge(adp_clean, on=["Player", "Position"], how="left")
    else:
        # If one of the inputs is empty, still return the projection data with missing ADP
        merged_df = proj_clean.copy()
        merged_df["ADP"] = pd.NA

    # Ensure a consistent FPTS column.  Some projection files use a different
    # name (e.g., FantasyPoints).  Look for columns that case-insensitively
    # match "FPTS" or "FANTASYPOINTS" and standardise to "FPTS".
    fpts_col = None
    for candidate in merged_df.columns:
        if candidate.lower() in {"fpts", "fantasypoints", "fantasy_points", "fpts"}:
            fpts_col = candidate
            break
    if fpts_col and fpts_col != "FPTS":
        merged_df = merged_df.rename(columns={fpts_col: "FPTS"})

    # Select only the relevant columns if they exist
    cols: List[str] = []
    for col in ["Player", "Team", "Position", "FPTS", "ADP"]:
        if col in merged_df.columns:
            cols.append(col)
    final_df = merged_df[cols].copy()

    # Drop rows where critical information is missing
    # Only attempt to drop FPTS if the column exists
    drop_subset = [c for c in ["Player", "Position", "FPTS"] if c in final_df.columns]
    final_df = final_df.dropna(subset=drop_subset)
    # Optionally fill missing ADP values with a placeholder
    if fill_missing_adp is not None and "ADP" in final_df.columns:
        final_df["ADP"] = final_df["ADP"].fillna(fill_missing_adp)

    # Reset index for cleanliness
    final_df = final_df.reset_index(drop=True)
    return final_df


def load_players_as_objects(data_dir: str, *, fill_missing_adp: float | None = 999) -> List[Player]:
    """Load and return a list of ``Player`` objects from raw data files.

    This convenience function wraps :func:`load_clean_player_data` and
    constructs :class:`draft_models.Player` instances for each row in the
    resulting DataFrame.  If the DataFrame lacks certain columns (e.g., team
    information), empty strings will be used in the corresponding fields of
    the ``Player`` objects.

    Parameters
    ----------
    data_dir : str
        Directory containing the projection and ADP CSV files.
    fill_missing_adp : float or None, optional
        Placeholder value to assign to missing ADP entries.

    Returns
    -------
    List[Player]
        A list of Player instances constructed from the cleaned data.
    """
    # Perform the same cleaning/merging logic as load_clean_player_data
    df = load_clean_player_data(data_dir, fill_missing_adp=fill_missing_adp)
    # Import here to avoid circular dependency at module load time
    from draft_models import Player
    players: List[Player] = []
    for _, row in df.iterrows():
        players.append(
            Player(
                name=row.get("Player", ""),
                team=row.get("Team", ""),
                position=str(row.get("Position", "")).upper(),
                fpts=float(row.get("FPTS", 0.0)),
                adp=float(row.get("ADP", 999.0)),
            )
        )
    return players


if __name__ == "__main__":
    # Simple CLI test: load data from a provided directory and print the head
    import argparse

    parser = argparse.ArgumentParser(description="Load and clean fantasy football data.")
    parser.add_argument("data_dir", help="Directory containing projection and ADP CSV files")
    parser.add_argument("--no-fill-adp", dest="fill_adp", action="store_false",
                        help="Do not fill missing ADP values with a placeholder")
    args = parser.parse_args()

    df = load_clean_player_data(args.data_dir, fill_missing_adp=(999 if args.fill_adp else None))
    print(df.head())