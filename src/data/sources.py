"""画像ソースの抽象化（ディレクトリ / ZIP直読み）。

96GB級のZIPを展開せずに扱えるよう、ZIP内の画像を遅延読み出しする。
DBの `images.filepath` には以下の形式で記録する（04_DB設計書のスキーマ変更は不要）:

    ディレクトリ: /Volumes/Extreme Pro/BigGAN/train/ai/xxx.png
    ZIP内:        /Volumes/Extreme Pro/genimage-...zip!train/ai/xxx.png

`open_image_bytes()` はどちらの形式のパスも透過的に読める。学習・評価時の
DataLoaderからはワーカーごとに `ZipSource` を開き直すこと（ZipFileオブジェクトは
プロセス間・スレッド間で共有しない）。
"""

from __future__ import annotations

import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

# ZIPパスと内部パスの区切り（Java の jar URL 表記に倣う）
ZIP_SEPARATOR = "!"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageEntry:
    """ソース内の1画像。"""

    relpath: str          # ソースルートからの相対パス（例: "train/ai/xxx.png"）
    size_bytes: int
    filepath: str         # DBに記録する一意なパス


class ImageSource(ABC):
    """ディレクトリとZIPを同じインタフェースで扱うための基底クラス。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    @abstractmethod
    def iter_entries(self) -> Iterator[ImageEntry]:
        """ソース内の画像ファイルを列挙する。"""

    @abstractmethod
    def open(self, entry: ImageEntry) -> IO[bytes]:
        """画像のバイトストリームを開く。"""

    def close(self) -> None:
        """保持しているハンドルを解放する。"""

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class DirectorySource(ImageSource):
    """展開済みディレクトリを走査するソース。"""

    def iter_entries(self) -> Iterator[ImageEntry]:
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield ImageEntry(
                    relpath=path.relative_to(self.root).as_posix(),
                    size_bytes=path.stat().st_size,
                    filepath=str(path),
                )

    def open(self, entry: ImageEntry) -> IO[bytes]:
        return (self.root / entry.relpath).open("rb")


class ZipSource(ImageSource):
    """ZIPを展開せずに走査・読み出しするソース。"""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._zf: zipfile.ZipFile | None = None

    @property
    def zipfile_handle(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.root)
        return self._zf

    def iter_entries(self) -> Iterator[ImageEntry]:
        # 中央ディレクトリを読むだけなので展開は発生しない
        for info in self.zipfile_handle.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # macOSで作られたZIPに混ざるリソースフォークを除外
            if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                continue
            yield ImageEntry(
                relpath=name,
                size_bytes=info.file_size,
                filepath=f"{self.root}{ZIP_SEPARATOR}{name}",
            )

    def open(self, entry: ImageEntry) -> IO[bytes]:
        return self.zipfile_handle.open(entry.relpath)

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None


def open_source(path: str | Path, source_type: str = "auto") -> ImageSource:
    """パスとタイプ指定から適切な `ImageSource` を返す。"""
    p = Path(path)
    if source_type == "auto":
        source_type = "zip" if p.suffix.lower() == ".zip" else "dir"
    if source_type == "zip":
        return ZipSource(p)
    if source_type == "dir":
        return DirectorySource(p)
    raise ValueError(f"未知のソースタイプです: {source_type!r}")


def split_zip_path(filepath: str) -> tuple[str, str] | None:
    """`...zip!inner/path.png` を (zipパス, 内部パス) に分解する。ZIP形式でなければ None。"""
    marker = f".zip{ZIP_SEPARATOR}"
    idx = filepath.lower().find(marker)
    if idx == -1:
        return None
    boundary = idx + len(marker)
    return filepath[: boundary - 1], filepath[boundary:]


def open_image_bytes(filepath: str, max_bytes: int | None = None) -> bytes:
    """DBに記録された `filepath` から画像バイト列を読む（ディレクトリ／ZIP両対応）。

    `max_bytes` を指定すると先頭その分だけ読む（ヘッダのみ読みたい場合に使う）。
    """
    parts = split_zip_path(filepath)
    if parts is None:
        with open(filepath, "rb") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)

    zip_path, inner = parts
    with zipfile.ZipFile(zip_path) as zf, zf.open(inner) as f:
        return f.read() if max_bytes is None else f.read(max_bytes)
