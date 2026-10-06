#!/usr/bin/env python3
"""Build the public replication archive from an explicit list of files."""

import hashlib
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ".gitignore", "README.md", "ppb_paper.qmd", "ppb_supplement.qmd",
    "references_zotero.bib", "apa.csl", "data/README.md", "data/inputs.json",
    "code/Project.toml", "code/Manifest.toml", "code/README.md",
    "code/colloff.jl", "code/colloff_wixted.jl", "code/empirical.jl",
    "code/figure_warp.jl", "code/figures.jl", "code/inference.jl",
    "code/invariance.jl", "code/make.jl", "code/simulation.jl",
    "code/simulation_gaps.jl", "code/tables.jl", "code/verify_numbers.jl",
    "code/fetch_data.py", "code/package_replication.py",
    "code/ppb.R", "code/check_ppb_r.jl",
    "figures/fg_advantages.pdf", "figures/fg_conceptual.pdf", "figures/fg_warp.pdf",
    "tables/tbl_empirical.md", "tables/tbl_quartiles.md",
    "tables/tbl_lineup_responses.md", "tables/tbl_lineup_sensitivity.md",
    "tables/tbl_lineup_paired.md", "tables/tbl_cw_conditions.md",
    "tables/tbl_cw_contrast.md", "tables/tbl_interval_coverage.md",
    "tables/tbl_invariance_contrasts.md",
]


def main():
    contents = {name: (ROOT / name).read_bytes() for name in sorted(FILES)}
    checksums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in contents.items())
    contents["SHA256SUMS"] = checksums.encode()
    destination = ROOT / "ppb_replication.zip"
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / destination.name
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 6, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                output.writestr(info, data)
        destination.write_bytes(archive.read_bytes())
    print(f"Built {destination.name}: {len(contents)} files; original datasets excluded.")
    print(f"SHA-256: {hashlib.sha256(destination.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
