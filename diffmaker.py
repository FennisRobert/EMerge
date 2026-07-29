#!/usr/bin/env python3
"""
git_diff_to_md.py
Generates a raw .diff file and a human-readable Markdown report comparing two git branches.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> str:
    """Executes a subprocess command and returns stdout."""
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing {' '.join(cmd)}:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)


def sanitize_filename(name: str) -> str:
    """Sanitizes branch names for safe file creation."""
    return re.sub(r'[/\\?%*:|"<>]', "_", name)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two git branches and output raw diff and readable Markdown files."
    )
    parser.add_argument("branch1", help="Base branch name (e.g., main)")
    parser.add_argument("branch2", help="Target branch name (e.g., feature-branch)")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional custom output Markdown filename",
        default=None,
    )
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="Do not wrap diff blocks in expandable <details> tags",
    )

    args = parser.parse_args()

    # Verify git repository
    run_cmd(["git", "rev-parse", "--is-inside-work-tree"])

    b1, b2 = args.branch1, args.branch2
    diff_target = f"{b1}..{b2}"

    # Verify branches exist
    run_cmd(["git", "rev-parse", "--verify", b1])
    run_cmd(["git", "rev-parse", "--verify", b2])

    print(f"Comparing `{b1}` -> `{b2}`...")

    # 1. Generate Raw Patch File
    raw_diff = run_cmd(["git", "diff", diff_target])
    safe_b1 = sanitize_filename(b1)
    safe_b2 = sanitize_filename(b2)
    
    diff_filename = f"diff_{safe_b1}_to_{safe_b2}.diff"
    Path(diff_filename).write_text(raw_diff, encoding="utf-8")
    print(f"Generated raw diff: {diff_filename}")

    # 2. Gather Diff Data for Markdown
    stat_output = run_cmd(["git", "diff", "--stat", diff_target])
    name_status = run_cmd(["git", "diff", "--name-status", diff_target])

    status_map = {
        "M": "Modified",
        "A": "Added",
        "D": "Deleted",
        "R": "Renamed",
        "C": "Copied",
        "T": "Type Changed",
    }

    parsed_files = []
    for line in name_status.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        st = parts[0]
        src = parts[1]
        dst = parts[2] if len(parts) > 2 else None
        parsed_files.append((st, src, dst))

    # 3. Build Markdown Report
    md = []
    md.append(f"# Git Diff: `{b1}` → `{b2}`\n")
    md.append(f"- **Base Branch:** `{b1}`")
    md.append(f"- **Compare Branch:** `{b2}`")
    md.append(f"- **Raw Diff File:** `{diff_filename}`\n")

    md.append("## Summary Statistics\n")
    md.append("```text")
    md.append(stat_output.strip())
    md.append("```\n")

    md.append("## Changed Files Table\n")
    md.append("| Status | File Path |")
    md.append("| :--- | :--- |")

    for st, src, dst in parsed_files:
        st_code = st[0]
        label = status_map.get(st_code, st_code)
        path_str = f"`{src}` → `{dst}`" if dst else f"`{src}`"
        md.append(f"| {label} | {path_str} |")
    
    md.append("\n---\n")
    md.append("## Detailed File Changes\n")

    for st, src, dst in parsed_files:
        file_path = dst if dst else src
        st_code = st[0]
        label = status_map.get(st_code, st_code)

        title_path = f"`{src}` → `{dst}`" if dst else f"`{src}`"
        md.append(f"### {title_path}\n")
        md.append(f"**Status:** `{label}`\n")

        # Get diff for individual file
        file_diff = run_cmd(["git", "diff", diff_target, "--", file_path])
        
        if not file_diff.strip() and dst:
            # Handle renamed files without content changes
            file_diff = run_cmd(["git", "diff", diff_target, "--", src, dst])

        if file_diff.strip():
            if not args.no_collapse:
                md.append("<details><summary>Click to view diff</summary>\n")
            
            md.append("```diff")
            md.append(file_diff.strip())
            md.append("```\n")

            if not args.no_collapse:
                md.append("</details>\n")
        else:
            md.append("*No text changes (binary file or empty diff).*\n")

    md_filename = args.output if args.output else f"diff_{safe_b1}_to_{safe_b2}.md"
    Path(md_filename).write_text("\n".join(md), encoding="utf-8")
    print(f"Generated Markdown report: {md_filename}")


if __name__ == "__main__":
    main()