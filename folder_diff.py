#!/usr/bin/env python3
"""
folder_diff.py - confronta due cartelle e trova i file presenti in una e non nell'altra.

Due modalità di confronto:
  name  (default)  confronta per nome file (o per percorso relativo con --relpath)
  hash             confronta per contenuto (SHA-1): robusto se i file sono stati
                   rinominati o riesportati con nomi diversi tra le versioni

Esempi:
    python folder_diff.py dataset_v1/good dataset_v2/good
    python folder_diff.py dataset_v1/good dataset_v2/good --mode hash
    python folder_diff.py dataset_v1/good dataset_v2/good --out diff_v1_v2 --copy

Con --out vengono scritti only_in_A.txt e only_in_B.txt (A = prima cartella,
B = seconda cartella). Con --copy i file vengono anche copiati in
<out>/only_in_A/ e <out>/only_in_B/ per ispezionarli a colpo d'occhio.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_files(root: Path, recursive: bool, exts: set[str] | None) -> list[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in it if p.is_file()]
    if exts is not None:
        files = [p for p in files if p.suffix.lower() in exts]
    return sorted(files)


def sha1_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_index(
    root: Path,
    recursive: bool,
    exts: set[str] | None,
    mode: str,
    use_relpath: bool,
) -> dict[str, list[Path]]:
    """Mappa chiave di confronto -> lista di file con quella chiave."""
    files = list_files(root, recursive, exts)
    index: dict[str, list[Path]] = {}
    for i, p in enumerate(files, 1):
        if mode == "hash":
            key = sha1_of(p)
            if i % 500 == 0:
                print(f"  [{root}] hash {i}/{len(files)}", file=sys.stderr)
        elif use_relpath:
            key = p.relative_to(root).as_posix()
        else:
            key = p.name
        index.setdefault(key, []).append(p)
    return index


def only_in(first: dict[str, list[Path]], second: dict[str, list[Path]]) -> list[Path]:
    """File di `first` la cui chiave non compare in `second`."""
    return sorted(p for k, paths in first.items() if k not in second for p in paths)


def write_list(path: Path, root: Path, files: list[Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(p.relative_to(root).as_posix() + "\n")


def copy_files(dest: Path, root: Path, files: list[Path]) -> None:
    for p in files:
        target = dest / p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def print_section(title: str, root: Path, files: list[Path], limit: int) -> None:
    print(f"\n{title}: {len(files)} file")
    for p in files[:limit]:
        print(f"  {p.relative_to(root).as_posix()}")
    if len(files) > limit:
        print(f"  ... altri {len(files) - limit} (usa --out per la lista completa)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Trova i file presenti in una cartella e non nell'altra."
    )
    ap.add_argument("dir_a", type=Path, help="prima cartella (A)")
    ap.add_argument("dir_b", type=Path, help="seconda cartella (B)")
    ap.add_argument(
        "--mode",
        choices=["name", "hash"],
        default="name",
        help="confronto per nome (default) o per contenuto",
    )
    ap.add_argument(
        "--relpath",
        action="store_true",
        help="in modalità name usa il percorso relativo invece del solo nome file",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="non scendere nelle sottocartelle",
    )
    ap.add_argument(
        "--all-files",
        action="store_true",
        help="considera tutti i file, non solo le immagini",
    )
    ap.add_argument(
        "--ext",
        nargs="+",
        help="estensioni da considerare (es. .png .bmp); sostituisce il default immagini",
    )
    ap.add_argument("--out", type=Path, help="cartella dove salvare le liste (e le copie)")
    ap.add_argument(
        "--copy",
        action="store_true",
        help="con --out, copia anche i file in only_in_A/ e only_in_B/",
    )
    ap.add_argument("--limit", type=int, default=50, help="righe da stampare per sezione")
    args = ap.parse_args()

    for d in (args.dir_a, args.dir_b):
        if not d.is_dir():
            ap.error(f"cartella non trovata: {d}")
    if args.copy and not args.out:
        ap.error("--copy richiede --out")

    if args.all_files:
        exts = None
    elif args.ext:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}
    else:
        exts = IMAGE_EXTS

    recursive = not args.no_recursive

    print(f"A: {args.dir_a.resolve()}")
    print(f"B: {args.dir_b.resolve()}")
    print(f"Confronto per: {args.mode}")

    index_a = build_index(args.dir_a, recursive, exts, args.mode, args.relpath)
    index_b = build_index(args.dir_b, recursive, exts, args.mode, args.relpath)

    n_a = sum(len(v) for v in index_a.values())
    n_b = sum(len(v) for v in index_b.values())
    print(f"\nFile in A: {n_a} | File in B: {n_b} | In comune: {len(set(index_a) & set(index_b))}")

    only_a = only_in(index_a, index_b)
    only_b = only_in(index_b, index_a)

    print_section("Solo in A", args.dir_a, only_a, args.limit)
    print_section("Solo in B", args.dir_b, only_b, args.limit)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        write_list(args.out / "only_in_A.txt", args.dir_a, only_a)
        write_list(args.out / "only_in_B.txt", args.dir_b, only_b)
        if args.copy:
            copy_files(args.out / "only_in_A", args.dir_a, only_a)
            copy_files(args.out / "only_in_B", args.dir_b, only_b)
        print(f"\nRisultati salvati in {args.out.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())