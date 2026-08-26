from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOTS = (
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "tests",
    ROOT / "benchmarks",
    ROOT / "paper",
    ROOT / "freetoken-kernel-cache",
    ROOT / "python/freetoken/daemon",
)
FUNCTIONAL_REPOSITORY = "https://github.com/agisota/freetoken-mlx.git"
UPSTREAM_REPOSITORY = "https://github.com/FlashML-org/FreeToken"
FUNCTIONAL_REPOSITORY_WEB = FUNCTIONAL_REPOSITORY.removesuffix(".git")
FUNCTIONAL_GITHUB_URL = re.compile(r"https://github\.com/[^\s)<]*[Ff]ree[Tt]oken[^\s)<]*")
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
FENCE = re.compile(r"^\s*```", re.MULTILINE)
SCRIPT_PATH = re.compile(r"scripts/[A-Za-z0-9_./-]+\.sh")


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    for directory in DOC_ROOTS[1:]:
        files.extend(directory.rglob("*.md"))
    return sorted(path for path in files if path.is_file())


def _cli_commands() -> set[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "freetoken.cli", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return set(re.findall(r"^  ([a-z][a-z-]+)\s+", completed.stdout, flags=re.MULTILINE))


class DocumentationContractTest(unittest.TestCase):
    def test_documented_cli_commands_match_dispatcher(self) -> None:
        cli_doc = (ROOT / "docs/cli.md").read_text(encoding="utf-8")
        table = cli_doc.split("## Команды верхнего уровня", 1)[1].split("## Проверяемые поверхности команд", 1)[0]
        documented = set(re.findall(r"^\| `([^`]+)` \|", table, flags=re.MULTILINE))
        self.assertSetEqual(documented, _cli_commands())

    def test_relative_markdown_links_resolve_and_fences_are_balanced(self) -> None:
        for document in _markdown_files():
            text = document.read_text(encoding="utf-8")
            self.assertEqual(
                len(FENCE.findall(text)) % 2,
                0,
                f"unbalanced fenced code block in {document.relative_to(ROOT)}",
            )
            for raw_target in MARKDOWN_LINK.findall(text):
                target = raw_target.strip().strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:", "data:")):
                    continue
                target = unquote(target.split("#", 1)[0])
                if not target:
                    continue
                self.assertTrue(
                    (document.parent / target).exists(),
                    f"broken link in {document.relative_to(ROOT)}: {raw_target}",
                )

    def test_repository_urls_and_upstream_attribution_are_scoped(self) -> None:
        for document in _markdown_files():
            text = document.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "git clone https://github.com/" in line:
                    self.assertIn(FUNCTIONAL_REPOSITORY, line)
                if UPSTREAM_REPOSITORY in line:
                    self.assertEqual(document, ROOT / "README.md")
            for url in FUNCTIONAL_GITHUB_URL.findall(text):
                if url != UPSTREAM_REPOSITORY:
                    self.assertTrue(url.startswith(FUNCTIONAL_REPOSITORY_WEB), url)

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'Homepage = "{FUNCTIONAL_REPOSITORY_WEB}"', pyproject)
        self.assertIn(f'Repository = "{FUNCTIONAL_REPOSITORY_WEB}"', pyproject)
        self.assertIn(f'Issues = "{FUNCTIONAL_REPOSITORY_WEB}/issues"', pyproject)

    def test_workflow_script_references_exist(self) -> None:
        for workflow in (ROOT / ".github/workflows").glob("*.yml"):
            for script in SCRIPT_PATH.findall(workflow.read_text(encoding="utf-8")):
                self.assertTrue((ROOT / script).is_file(), f"missing {script} referenced by {workflow.name}")
