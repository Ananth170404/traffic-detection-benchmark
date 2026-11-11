from tdb.data.yolo import Box, iter_split, parse_label_file, source_video


def test_parse_label_file_skips_malformed_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("4 0.5 0.5 0.1 0.2\n\n3 0.1\n11 0.1 0.2 0.3 0.4\n")
    assert parse_label_file(f) == [Box(4, 0.5, 0.5, 0.1, 0.2), Box(11, 0.1, 0.2, 0.3, 0.4)]


def test_source_video():
    name = "siang_15112021_1_mp4-95_jpg.rf.786a85df12823ffad7d46a163b2adf85.jpg"
    assert source_video(name) == "siang_15112021_1"
    assert source_video("adit_mp4-77_jpg.rf.767fb3.jpg") == "adit"


def test_iter_split_handles_missing_label(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir()
    (tmp_path / "train" / "images" / "a.jpg").write_bytes(b"")
    (tmp_path / "train" / "images" / "b.jpg").write_bytes(b"")
    (tmp_path / "train" / "labels" / "a.txt").write_text("4 0.5 0.5 0.1 0.1\n")
    items = list(iter_split(tmp_path, "train"))
    assert [p.name for p, _ in items] == ["a.jpg", "b.jpg"]
    assert len(items[0][1]) == 1 and items[1][1] == []
