from pathlib import PosixPath


def check_file(file_path: PosixPath):
    if not file_path.is_file():
        raise FileNotFoundError(f"Can not find necessary file: {file_path}")
    else:
        return file_path
