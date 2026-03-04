import os

from note_logger import append_markdown_entry


def test_append_markdown_entry_creates_file(tmp_path):
    path = tmp_path / "notes.md"
    result = append_markdown_entry("Hello world", notes_path=str(path))
    assert result == str(path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "## " in content
    assert "Hello world" in content


def test_append_markdown_entry_appends(tmp_path):
    path = tmp_path / "notes.md"
    append_markdown_entry("First", notes_path=str(path))
    append_markdown_entry("Second", notes_path=str(path))
    content = path.read_text(encoding="utf-8")
    assert content.count("## ") == 2
    assert "First" in content and "Second" in content

