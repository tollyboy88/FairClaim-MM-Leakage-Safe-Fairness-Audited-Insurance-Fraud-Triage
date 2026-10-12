"""Refresh the two manuscript DOCX entries in the submission archive."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "AOA_submission_package.zip"
REPLACEMENTS = {
    "AOA_submission_package/delivery/Revised_Manuscript_Clean.docx":
        ROOT / "publication/aoa/delivery/Revised_Manuscript_Clean.docx",
    "AOA_submission_package/delivery/Revised_Manuscript_Changes_Highlighted.docx":
        ROOT / "publication/aoa/delivery/Revised_Manuscript_Changes_Highlighted.docx",
}


def main() -> None:
    with NamedTemporaryFile(suffix=".zip", delete=False, dir=ROOT) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(ARCHIVE, "r") as source, zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for item in source.infolist():
                if item.filename in REPLACEMENTS:
                    target.writestr(item, REPLACEMENTS[item.filename].read_bytes())
                else:
                    target.writestr(item, source.read(item.filename))
        os.replace(temporary, ARCHIVE)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
