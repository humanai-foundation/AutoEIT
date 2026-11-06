from src.transcription.resegmented_items import iter_resegmented_items


def test_discovers_and_sorts_items(monkeypatch, tmp_path):
    folder = tmp_path / "version_a"
    folder.mkdir()

    filenames = [
        "recording_b_item2.mp3",
        "recording_a_item10.mp3",
        "recording_a_item2.mp3",
        "recording_a_item0.mp3",
        "invalid_filename.mp3",
        "recording_a_item1.wav",
    ]

    for filename in filenames:
        (folder / filename).touch()

    monkeypatch.setattr(
        "src.transcription.resegmented_items.data_path",
        lambda *parts: folder,
    )

    items = iter_resegmented_items("version_a")

    assert [(recording, index) for recording, index, _ in items] == [
        ("recording_a", 0),
        ("recording_a", 2),
        ("recording_a", 10),
        ("recording_b", 2),
    ]
    assert all(path.suffix == ".mp3" for _, _, path in items)


def test_accepts_attention_prefix(monkeypatch, tmp_path):
    folder = tmp_path / "version_b"
    folder.mkdir()
    audio_file = folder / "ATTENTION_recording_a_item0.mp3"
    audio_file.touch()

    monkeypatch.setattr(
        "src.transcription.resegmented_items.data_path",
        lambda *parts: folder,
    )

    items = iter_resegmented_items("version_b")

    assert len(items) == 1
    assert items[0][0] == "recording_a"
    assert items[0][1] == 0
    assert items[0][2] == audio_file


def test_empty_directory_returns_empty_list(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.transcription.resegmented_items.data_path",
        lambda *parts: tmp_path,
    )

    assert iter_resegmented_items("version_a") == []


def test_uses_requested_version_directory(monkeypatch, tmp_path):
    requested_paths = []

    def fake_data_path(*parts):
        requested_paths.append(parts)
        return tmp_path

    monkeypatch.setattr(
        "src.transcription.resegmented_items.data_path",
        fake_data_path,
    )

    iter_resegmented_items("version_b")

    assert requested_paths == [("segmented", "v2", "version_b")]
