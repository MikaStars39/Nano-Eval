from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nanoeval.utils.args import parse_task_names, parse_task_pass_k


class TestArgs(unittest.TestCase):
    def test_parse_task_names_supports_jsonl_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            (task_dir / "aime2024.jsonl").write_text("{}", encoding="utf-8")

            names = parse_task_names(tasks_arg="aime2024.jsonl", task_dir=task_dir)
            self.assertEqual(names, ["aime2024"])

    def test_parse_task_names_supports_task_spec_with_pass_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            (task_dir / "aime2024.jsonl").write_text("{}", encoding="utf-8")

            names = parse_task_names(tasks_arg="aime2024@4", task_dir=task_dir)
            self.assertEqual(names, ["aime2024"])

    def test_parse_task_pass_k_supports_per_task_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            (task_dir / "aime2024.jsonl").write_text("{}", encoding="utf-8")
            (task_dir / "aime2025.jsonl").write_text("{}", encoding="utf-8")

            names, pass_k_by_task = parse_task_pass_k(
                tasks_arg="aime2024@4,aime2025@8",
                task_dir=task_dir,
                default_pass_k=1,
            )

            self.assertEqual(names, ["aime2024", "aime2025"])
            self.assertEqual(pass_k_by_task["aime2024"], 4)
            self.assertEqual(pass_k_by_task["aime2025"], 8)

    def test_parse_task_pass_k_uses_default_when_not_specified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            (task_dir / "aime2024.jsonl").write_text("{}", encoding="utf-8")
            (task_dir / "aime2025.jsonl").write_text("{}", encoding="utf-8")

            names, pass_k_by_task = parse_task_pass_k(
                tasks_arg="aime2024,aime2025@8",
                task_dir=task_dir,
                default_pass_k=3,
            )

            self.assertEqual(names, ["aime2024", "aime2025"])
            self.assertEqual(pass_k_by_task["aime2024"], 3)
            self.assertEqual(pass_k_by_task["aime2025"], 8)


if __name__ == "__main__":
    unittest.main()
