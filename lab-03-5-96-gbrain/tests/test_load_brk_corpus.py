"""Offline tests for load_brk_corpus.build_pages (pure transform, no GBrain).

Runs against the REAL W2.7 corpus when present, plus synthetic edge cases.
"""
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from load_brk_corpus import build_pages  # noqa: E402

_BRK = pathlib.Path(os.path.expanduser(
    "~/code/agent-prep/lab-02-7-pageindex/data/brk_corpus.json"))


def test_slug_and_content_shape_has_frontmatter_title():
    pages = build_pages([{"id": "brk_0002", "title": "Table of Contents", "text": "BERKSHIRE ..."}])
    assert pages[0][0] == "sections/brk_0002"
    assert pages[0][1] == '---\ntitle: "Table of Contents"\n---\n\n# Table of Contents\n\nBERKSHIRE ...\n'


def test_breadcrumb_title_keeps_distinctive_tail():
    # the shared "Berkshire ... Annual Report >" prefix must be dropped
    pages = build_pages([{"id": "x", "title": "Berkshire Hathaway 2023 Annual Report > Chairman's Letter", "text": "body"}])
    assert 'title: "Chairman\'s Letter"' in pages[0][1]
    assert "Annual Report" not in pages[0][1].split("\n---")[0]  # not in frontmatter title


def test_custom_prefix():
    pages = build_pages([{"id": "x1", "title": "T", "text": "body"}], prefix="tenk")
    assert pages[0][0] == "tenk/x1"


def test_skips_empty_id_or_text():
    corpus = [
        {"id": "", "title": "no id", "text": "x"},
        {"id": "ok", "title": "t", "text": ""},
        {"id": "good", "title": "t", "text": "real"},
    ]
    assert [s for s, _ in build_pages(corpus)] == ["sections/good"]


def test_missing_title_falls_back_to_slug_id():
    pages = build_pages([{"id": "x", "text": "body only"}])
    assert 'title: "x"' in pages[0][1]  # fallback to id when no breadcrumb


def test_quotes_in_title_escaped():
    pages = build_pages([{"id": "x", "title": 'The "Big" Letter', "text": "b"}])
    assert 'title: "The \\"Big\\" Letter"' in pages[0][1]


@pytest.mark.skipif(not _BRK.exists(), reason="W2.7 brk_corpus.json not present")
def test_real_corpus_loads_and_slugs_are_unique():
    corpus = json.loads(_BRK.read_text())
    pages = build_pages(corpus)
    assert len(pages) >= 40  # 44 sections, minus any empty
    slugs = [s for s, _ in pages]
    assert len(slugs) == len(set(slugs))  # no slug collisions
    assert all(s.startswith("sections/") for s in slugs)
    assert all(content.strip() for _, content in pages)  # no empty bodies
