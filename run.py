"""
Top-level runner: refresh → train → export for all (or selected) leagues.

Usage:
    python run.py                          # all leagues, current season only
    python run.py pl laliga               # subset of leagues
    python run.py --bootstrap             # download full history first
    python run.py --bootstrap bundesliga  # bootstrap one league
    python run.py --export-only <path>    # re-export without retraining

Requires the web repo path to be set via WEB_ROOT env var or passed after flags:
    WEB_ROOT=../football_predictor_web python run.py
    python run.py --web ../football_predictor_web
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from src.leagues import LEAGUES
from src.refresh_data import bootstrap, refresh
from src.train import train_and_evaluate
from src.export_web import main as export_main


def _parse_args() -> tuple[list[str], bool, Path | None, bool]:
    args = sys.argv[1:]
    do_bootstrap = "--bootstrap" in args
    export_only = "--export-only" in args
    args = [a for a in args if a not in ("--bootstrap", "--export-only")]

    web_root: Path | None = None
    if "--web" in args:
        idx = args.index("--web")
        if idx + 1 < len(args):
            web_root = Path(args[idx + 1]).resolve()
            args = args[:idx] + args[idx + 2:]

    if web_root is None:
        env = os.environ.get("WEB_ROOT")
        if env:
            web_root = Path(env).resolve()

    league_ids = [a for a in args if a in LEAGUES]
    if not league_ids:
        league_ids = list(LEAGUES.keys())

    return league_ids, do_bootstrap, web_root, export_only


def main() -> None:
    league_ids, do_bootstrap, web_root, export_only = _parse_args()

    print(f"Leagues: {', '.join(league_ids)}")

    if not export_only:
        # 1. Refresh / bootstrap
        for lid in league_ids:
            league = LEAGUES[lid]
            if do_bootstrap:
                print(f"\nBootstrapping {league.name}…")
                bootstrap(league)
            else:
                print(f"\nRefreshing {league.name}…")
                updated = refresh(league)
                if updated:
                    print(f"  Updated seasons: {', '.join(updated)}")
                else:
                    print(f"  Already up to date")

        # 2. Train
        print("\nTraining models…")
        for lid in league_ids:
            league = LEAGUES[lid]
            print(f"  {league.name}…", end=" ", flush=True)
            try:
                result = train_and_evaluate(lid)
                m = result["metrics"]
                print(f"accuracy={m['model_accuracy']:.1%}  log-loss={m['model_log_loss']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")

    # 3. Export
    if web_root is None:
        print("\nNo web root specified — skipping export.")
        print("Set WEB_ROOT env var or pass --web <path>")
        return

    print(f"\nExporting to {web_root}…")
    export_main(web_root, league_ids)
    print("\nDone.")


if __name__ == "__main__":
    main()
