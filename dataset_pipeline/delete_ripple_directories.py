"""Delete generated ripple candidate directories for one or all subjects."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import subjects


DEFAULT_SESSIONS = ("Sleep1", "Sleep2")


def delete_ripple_directories(subject_id: str, sessions=DEFAULT_SESSIONS) -> int:
    """Delete each requested session's ``ripples`` directory."""
    subject_directory = Path(subjects.load(subject_id)["olm"]["directory"]).resolve()
    deleted = 0

    for session in sessions:
        if not session or session in {".", ".."} or Path(session).name != session:
            raise ValueError(f"Unsafe session name: {session!r}")

        expected_parent = (subject_directory / session).resolve()
        ripple_directory = (expected_parent / "ripples").resolve()
        if (
            expected_parent.parent != subject_directory
            or ripple_directory.parent != expected_parent
            or ripple_directory.name != "ripples"
        ):
            raise ValueError(f"Refusing unsafe deletion target: {ripple_directory}")

        if not ripple_directory.is_dir():
            print(f"Not found: {ripple_directory}")
            continue

        shutil.rmtree(ripple_directory)
        print(f"Deleted: {ripple_directory}")
        deleted += 1

    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subject_id",
        nargs="?",
        help="Subject ID from the configured subject metadata",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_subjects",
        help="Delete ripple directories for every configured subject",
    )
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    args = parser.parse_args()

    if bool(args.subject_id) == args.all_subjects:
        parser.error("provide either a subject ID or --all")

    subject_ids = subjects.get_subjects() if args.all_subjects else [args.subject_id]
    deleted = 0
    for subject_id in subject_ids:
        print(f"Subject: {subject_id}")
        deleted += delete_ripple_directories(subject_id, args.sessions)

    print(
        f"Deleted {deleted} ripple director{'y' if deleted == 1 else 'ies'} "
        f"across {len(subject_ids)} subject{'s' if len(subject_ids) != 1 else ''}."
    )


if __name__ == "__main__":
    main()
