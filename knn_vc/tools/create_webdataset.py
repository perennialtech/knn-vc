import argparse
from pathlib import Path

import webdataset as wds
from tqdm import tqdm


def prepare_shards(
    output_dir: str, audio_dir: str, ssl_dir: str, n_tars: int, ext: str
):
    """
    Creates multiple webdataset shards by pairing SSL features and audio files.
    """
    if n_tars < 1:
        raise ValueError(f"n_tars must be at least 1, got {n_tars}")

    output_path = Path(output_dir)
    audio_root = Path(audio_dir)
    ssl_root = Path(ssl_dir)
    normalized_ext = ext if ext.startswith(".") else f".{ext}"
    sample_ext = normalized_ext.lstrip(".")

    output_path.mkdir(parents=True, exist_ok=True)

    # gather all valid pairs
    file_pairs = []
    ssl_paths = sorted(ssl_root.rglob("*.pt"))

    for ssl_path in tqdm(ssl_paths, desc="Searching for file pairs..."):
        rel_path = ssl_path.relative_to(ssl_root)
        audio_path = (audio_root / rel_path).with_suffix(normalized_ext)

        if audio_path.exists():
            key = rel_path.with_suffix("").as_posix()
            file_pairs.append((key, ssl_path, audio_path))
        else:
            print(f"Warning: Audio missing for {ssl_path}")

    if not file_pairs:
        raise FileNotFoundError(
            f"No matching {sample_ext} audio files found for .pt files under {ssl_root}"
        )

    # create the pattern for filenames (e.g., shard-00001.tar)
    print(f"{len(file_pairs)} samples will be written across {n_tars} shards")
    writers = [
        wds.TarWriter(str(output_path / f"shard-{i:05d}.tar"))
        for i in range(n_tars)  # type: ignore
    ]

    # distribute samples across writers
    for i, (key, ssl_path, audio_path) in tqdm(
        enumerate(file_pairs), desc="Writing pairs"
    ):
        writer = writers[i % n_tars]

        with open(ssl_path, "rb") as f:
            ssl_data = f.read()
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # WebDataset uses the keys to group files into a single sample
        sample = {"__key__": key, "pt": ssl_data, sample_ext: audio_data}
        writer.write(sample)

    # close writers to finalize the tar files
    for writer in writers:
        writer.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create sharded WebDataset")
    parser.add_argument("output_dir", help="Directory to save .tar files")
    parser.add_argument("audio_dir")
    parser.add_argument("ssl_dir")
    parser.add_argument("n_tars", type=int, help="Number of shards to create")
    parser.add_argument("--ext", default=".flac", type=str)
    args = parser.parse_args(argv)

    prepare_shards(args.output_dir, args.audio_dir, args.ssl_dir, args.n_tars, args.ext)
    print(f"\nDone! Shards are located in: {args.output_dir}")


if __name__ == "__main__":
    main()
