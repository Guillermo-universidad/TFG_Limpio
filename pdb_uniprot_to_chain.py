#!/usr/bin/env python3
import argparse
import csv
import sys
import time
from typing import Iterable, List, Optional

import requests


def _first_matching_key(keys: Iterable[str], target_upper: str) -> Optional[str]:
    for key in keys:
        if key.upper() == target_upper:
            return key
    return None


def _get_json(
    url: str,
    session: requests.Session,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> Optional[dict]:
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.ok:
                return resp.json()
            if resp.status_code not in (429, 500, 502, 503, 504):
                return None
        except requests.RequestException:
            pass

        if attempt < max_retries:
            time.sleep(retry_sleep)

    return None


def get_chain_ids(
    pdb_id: str,
    uniprot_id: str,
    timeout: int = 10,
    session: Optional[requests.Session] = None,
    max_retries: int = 2,
    retry_sleep: float = 1.0,
) -> List[str]:
    """
    Return chain IDs for a (PDB ID, UniProt ID) pair using PDBe mappings.
    Tries in order: mapping by PDB, best_structures, all_structures.
    """
    pdb_id_norm = pdb_id.strip().lower()
    uniprot_norm = uniprot_id.strip().upper()
    chain_ids = set()

    # 1) Mapping by PDB -> UniProt
    session = session or requests.Session()

    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id_norm}"
    data = _get_json(url, session, timeout, max_retries, retry_sleep)
    if data:
        pdb_data = data.get(pdb_id_norm, {})
        uniprot_map = pdb_data.get("UniProt", {})
        match_key = _first_matching_key(uniprot_map.keys(), uniprot_norm)
        if match_key:
            mappings = uniprot_map.get(match_key, {}).get("mappings", [])
            for mapping in mappings:
                chain_id = mapping.get("chain_id") or mapping.get("struct_asym_id")
                if chain_id:
                    chain_ids.add(chain_id)

    # 2) best_structures by UniProt
    if not chain_ids:
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot_norm}"
        data = _get_json(url, session, timeout, max_retries, retry_sleep)
        if data:
            entries = data.get(uniprot_norm, [])
            for entry in entries:
                if entry.get("pdb_id", "").lower() == pdb_id_norm:
                    chain_id = entry.get("chain_id")
                    if chain_id:
                        chain_ids.add(chain_id)

    # 3) all_structures by UniProt
    if not chain_ids:
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/all_structures/{uniprot_norm}"
        data = _get_json(url, session, timeout, max_retries, retry_sleep)
        if data:
            entries = data.get(uniprot_norm, {})
            pdb_entry = entries.get(pdb_id_norm, {})
            for chain_id in pdb_entry.get("chains", []):
                if chain_id:
                    chain_ids.add(chain_id)

    return sorted(chain_ids)


def get_primary_chain(
    pdb_id: str,
    uniprot_id: str,
    timeout: int = 10,
    session: Optional[requests.Session] = None,
    max_retries: int = 2,
    retry_sleep: float = 1.0,
) -> Optional[str]:
    """Return a single chain, prioritizing PDB->UniProt mapping when available."""
    pdb_id_norm = pdb_id.strip().lower()
    uniprot_norm = uniprot_id.strip().upper()

    # 1) Mapping by PDB -> UniProt (most specific)
    session = session or requests.Session()

    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id_norm}"
    data = _get_json(url, session, timeout, max_retries, retry_sleep)
    if data:
        pdb_data = data.get(pdb_id_norm, {})
        uniprot_map = pdb_data.get("UniProt", {})
        match_key = _first_matching_key(uniprot_map.keys(), uniprot_norm)
        if match_key:
            mappings = uniprot_map.get(match_key, {}).get("mappings", [])
            for mapping in mappings:
                chain_id = mapping.get("chain_id") or mapping.get("struct_asym_id")
                if chain_id:
                    return chain_id

    # 2) best_structures by UniProt
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot_norm}"
    data = _get_json(url, session, timeout, max_retries, retry_sleep)
    if data:
        entries = data.get(uniprot_norm, [])
        for entry in entries:
            if entry.get("pdb_id", "").lower() == pdb_id_norm:
                chain_id = entry.get("chain_id")
                if chain_id:
                    return chain_id

    # 3) Fallback to any mapping result
    chains = get_chain_ids(
        pdb_id,
        uniprot_id,
        timeout=timeout,
        session=session,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    if chains:
        return chains[0]

    return None


def process_csv(
    input_csv: str,
    output_csv: str,
    pdb_col: str = "pdb_id",
    uniprot_col: str = "uniprot_id",
    fallback_uniprot_col: Optional[str] = None,
    delimiter: str = ",",
    timeout: int = 10,
    limit: Optional[int] = None,
    progress_every: int = 100,
    single_chain: bool = False,
    resume: bool = False,
    sleep: float = 0.0,
    max_retries: int = 2,
    retry_sleep: float = 1.0,
) -> None:
    processed_rows = 0
    if resume:
        try:
            with open(output_csv, "r", newline="") as f_prev:
                prev_reader = csv.DictReader(f_prev, delimiter=delimiter)
                for _ in prev_reader:
                    processed_rows += 1
        except FileNotFoundError:
            processed_rows = 0

    open_mode = "a" if resume and processed_rows > 0 else "w"
    with open(input_csv, "r", newline="") as f_in, open(output_csv, open_mode, newline="") as f_out:
        reader = csv.DictReader(f_in, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("CSV sin cabecera")

        fieldnames = list(reader.fieldnames)
        if "chain_id" not in fieldnames:
            fieldnames.append("chain_id")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=delimiter)
        if open_mode == "w":
            writer.writeheader()

        session = requests.Session()

        for idx, row in enumerate(reader, start=1):
            if limit is not None and idx > limit:
                break

            if processed_rows and idx <= processed_rows:
                continue

            pdb_id = (row.get(pdb_col) or "").strip()
            uniprot_id = (row.get(uniprot_col) or "").strip()
            fallback_uniprot_id = ""
            if fallback_uniprot_col:
                fallback_uniprot_id = (row.get(fallback_uniprot_col) or "").strip()

            chains: List[str] = []
            if pdb_id and uniprot_id:
                if single_chain:
                    primary = get_primary_chain(
                        pdb_id,
                        uniprot_id,
                        timeout=timeout,
                        session=session,
                        max_retries=max_retries,
                        retry_sleep=retry_sleep,
                    )
                    if primary:
                        chains = [primary]
                else:
                    chains = get_chain_ids(
                        pdb_id,
                        uniprot_id,
                        timeout=timeout,
                        session=session,
                        max_retries=max_retries,
                        retry_sleep=retry_sleep,
                    )
            if not chains and pdb_id and fallback_uniprot_id:
                if single_chain:
                    primary = get_primary_chain(
                        pdb_id,
                        fallback_uniprot_id,
                        timeout=timeout,
                        session=session,
                        max_retries=max_retries,
                        retry_sleep=retry_sleep,
                    )
                    if primary:
                        chains = [primary]
                else:
                    chains = get_chain_ids(
                        pdb_id,
                        fallback_uniprot_id,
                        timeout=timeout,
                        session=session,
                        max_retries=max_retries,
                        retry_sleep=retry_sleep,
                    )

            row["chain_id"] = ";".join(chains)
            writer.writerow(row)

            if sleep > 0:
                time.sleep(sleep)

            if progress_every > 0 and idx % progress_every == 0:
                print(f"Procesadas {idx} filas...", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrae chain_id a partir de un PDB_id y UniProt ID (PDBe)."
    )
    parser.add_argument("--pdb", help="PDB ID (ej: 1A8M)")
    parser.add_argument("--uniprot", help="UniProt ID (ej: P12345)")
    parser.add_argument("--csv-in", dest="csv_in", help="CSV de entrada")
    parser.add_argument("--csv-out", dest="csv_out", help="CSV de salida")
    parser.add_argument("--pdb-col", default="pdb_id", help="Nombre columna PDB")
    parser.add_argument("--uniprot-col", default="uniprot_id", help="Nombre columna UniProt")
    parser.add_argument("--delimiter", default=",", help="Separador CSV (por defecto ,)")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout HTTP en segundos")
    parser.add_argument("--limit", type=int, help="Procesar solo las primeras N filas")
    parser.add_argument(
        "--fallback-uniprot-col",
        help="Columna alternativa para UniProt (ej: accession)",
    )
    parser.add_argument(
        "--single-chain",
        action="store_true",
        help="Devuelve una sola cadena priorizando best_structures",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda usando el CSV de salida existente",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Pausa entre filas")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Reintentos por request HTTP",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="Espera entre reintentos HTTP",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Imprime progreso cada N filas",
    )

    args = parser.parse_args()

    if args.csv_in:
        if not args.csv_out:
            args.csv_out = args.csv_in.rsplit(".", 1)[0] + "_with_chain.csv"
        process_csv(
            args.csv_in,
            args.csv_out,
            pdb_col=args.pdb_col,
            uniprot_col=args.uniprot_col,
            fallback_uniprot_col=args.fallback_uniprot_col,
            delimiter=args.delimiter,
            timeout=args.timeout,
            limit=args.limit,
            progress_every=args.progress_every,
            single_chain=args.single_chain,
            resume=args.resume,
            sleep=args.sleep,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        print(f"OK: {args.csv_out}")
        return 0

    if not args.pdb or not args.uniprot:
        parser.print_help()
        return 2

    chains = get_chain_ids(args.pdb, args.uniprot, timeout=args.timeout)
    if chains:
        print(";".join(chains))
        return 0

    print("No se encontraron cadenas")
    return 1


if __name__ == "__main__":
    sys.exit(main())
