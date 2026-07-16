# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Focused regression tests for optional PGF figure export."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pytest

import benchmarks.plotting.export as figure_export

pytestmark = pytest.mark.benchmarks


class _BrokenPgfFigure:
    def __init__(self) -> None:
        self.attempts: list[Path] = []

    def savefig(self, path: Path, **_kwargs: object) -> None:
        output = Path(path)
        self.attempts.append(output)
        output.write_text("partial figure")
        if output.suffix == ".pgf":
            raise FileNotFoundError("pdflatex is unavailable")


def test_broken_tex_disables_pgf_once_without_losing_pdf_or_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(figure_export, "_pgf_available", None)
    fake_figure = _BrokenPgfFigure()
    figure = cast(plt.Figure, fake_figure)

    with pytest.warns(UserWarning, match="disabling optional PGF export") as warnings:
        figure_export.save_figure(figure, tmp_path, "first")
        figure_export.save_figure(figure, tmp_path, "second")

    assert len(warnings) == 1
    assert [path.name for path in fake_figure.attempts] == [
        "first.pdf",
        "first.png",
        "first.pgf",
        "second.pdf",
        "second.png",
    ]
    for name in ("first.pdf", "first.png", "second.pdf", "second.png"):
        assert (tmp_path / name).is_file()
    assert not (tmp_path / "first.pgf").exists()
    assert not (tmp_path / "second.pgf").exists()
