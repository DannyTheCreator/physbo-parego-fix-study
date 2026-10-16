"""Shared plotting and table helpers for VLMOP2 reports."""
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def archive_existing(path):
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path.rename(path.with_name(f"{path.stem}_archive_{stamp}{path.suffix}"))


def write_table(path, rows):
    archive_existing(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, folder, stem):
    for extension in ("png", "pdf"):
        path = folder / f"{stem}.{extension}"
        archive_existing(path)
        if extension == "pdf":
            fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
        else:
            fig.savefig(path, dpi=300)
    plt.close(fig)
