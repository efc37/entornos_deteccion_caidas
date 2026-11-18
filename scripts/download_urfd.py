"""Descarga el UR Fall Detection Dataset (URFD) desde su web oficial.

Fuente: https://fenix.ur.edu.pl/mkepski/ds/uf.html
Licencia: CC BY-NC-SA 4.0 (uso académico no comercial). Citar:
  B. Kwolek, M. Kepski. Human fall detection on embedded platform using depth maps
  and wireless accelerometer. Computer Methods and Programs in Biomedicine 117(3),
  489-501, 2014.

Se descargan SOLO la cámara frontal (cam0) y las etiquetas por frame. Los datos
quedan en data/raw/urfd/ (carpeta ignorada por git: no se redistribuyen).

Uso (desde la raíz del repositorio):
    python scripts/download_urfd.py               # imágenes RGB (zip, ~2-3 GB)
    python scripts/download_urfd.py --format mp4  # vídeos mp4 (más ligeros)
    python scripts/download_urfd.py --only fall-01 adl-01   # prueba rápida
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


def download(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ya existe: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {dest.name}: {100 * done / total:5.1f}% de {total / 1e6:.0f} MB",
                              end="", flush=True)
            print()
            tmp.replace(dest)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"\n  error ({attempt}/{retries}) al descargar {url}: {e}")
            time.sleep(2 * attempt)
    if tmp.exists():
        tmp.unlink()
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["rgb", "mp4"], default="rgb",
                    help="rgb: zip de imágenes PNG (recomendado); mp4: vídeo")
    ap.add_argument("--only", nargs="*", help="descargar solo estas secuencias (p. ej. fall-01)")
    ap.add_argument("--keep-zip", action="store_true", help="no borrar los zip tras extraerlos")
    args = ap.parse_args()

    out = C.URFD_RAW
    out.mkdir(parents=True, exist_ok=True)
    base = C.URFD_BASE_URL

    print("Etiquetas por frame (CSV)...")
    for name in ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]:
        download(f"{base}/{name}", out / name)

    seqs = [f"fall-{i:02d}" for i in range(1, C.URFD_N_FALLS + 1)] + \
           [f"adl-{i:02d}" for i in range(1, C.URFD_N_ADLS + 1)]
    if args.only:
        seqs = [s for s in seqs if s in set(args.only)]

    failed = []
    for i, seq in enumerate(seqs, 1):
        print(f"[{i}/{len(seqs)}] {seq}")
        if args.format == "mp4":
            if not download(f"{base}/{seq}-cam0.mp4", out / f"{seq}-cam0.mp4"):
                failed.append(seq)
            continue
        target_dir = out / f"{seq}-cam0-rgb"
        if target_dir.exists() and any(target_dir.rglob("*.png")):
            print("  ya extraído")
            continue
        zpath = out / f"{seq}-cam0-rgb.zip"
        if not download(f"{base}/{seq}-cam0-rgb.zip", zpath):
            failed.append(seq)
            continue
        with zipfile.ZipFile(zpath) as z:
            z.extractall(target_dir)
        if not args.keep_zip:
            zpath.unlink()

    print("\nHecho." if not failed else f"\nFallaron: {failed} (vuelva a ejecutar el script)")
    print("Comprobación: python -m src.datasets --check")


if __name__ == "__main__":
    main()
