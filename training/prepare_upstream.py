import hashlib
import io
import json
from pathlib import Path
import sys
import sysconfig
import tarfile
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parent
    if Path(sys.prefix).resolve() != (root.parent / '.venv').resolve():
        raise RuntimeError('Execute com .venv/bin/python.')
    upstream = json.loads((root / 'upstream.json').read_text())
    revision = upstream['revision']
    source = Path(sys.prefix) / 'src'
    archive_url = f'https://codeload.github.com/ShiqiYu/libfacedetection.train/tar.gz/{revision}'
    with urlopen(archive_url, timeout=60) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as package:
        for member in package.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts or parts[0] != 'yunet_train' or not member.isfile():
                continue
            destination = source.joinpath(*parts)
            if not destination.resolve().is_relative_to(source.resolve()):
                raise RuntimeError('Caminho upstream invalido.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(package.extractfile(member).read())
    site = Path(sysconfig.get_path('purelib'))
    (site / 'yunet_train.pth').write_text('../../../src\n')
    checkpoint = root / 'checkpoints' / 'yunet_n.pth'
    if not checkpoint.exists():
        url = f'https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/{revision}/weights/yunet_n.pth'
        with urlopen(url, timeout=60) as response:
            payload = response.read()
    else:
        payload = checkpoint.read_bytes()
    blob = b'blob ' + str(len(payload)).encode() + b'\0' + payload
    if hashlib.sha1(blob).hexdigest() != upstream['checkpoint_git_blob']:
        raise RuntimeError('Checkpoint diverge do blob oficial fixado.')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.write_bytes(payload)
    print(f'Pacote oficial e checkpoint verificados: {revision}')


if __name__ == '__main__':
    main()
