from pathlib import Path

from printvision.data import PrintFrameDataset


def test_manifest_schema(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "frames.csv").write_text("image,label,split\na.png,normal,train\n")
    # Schema validation is independent of image decoding; image errors occur on access.
    dataset = PrintFrameDataset(tmp_path / "frames.csv", tmp_path / "images", split="train")
    assert len(dataset) == 1
