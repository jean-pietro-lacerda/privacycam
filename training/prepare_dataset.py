import argparse
import hashlib
import http.client
import io
import json
import math
import random
import statistics
import zlib
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from zipfile import ZipFile

import cv2
import numpy as np
from yunet_train.tasks.face import parse_labelv2_file


REVISION = 'dca340aa082c71081a68d17db8e58b33a58a914b'
DATA_REVISION = '833d07e7bf3860f294242312fe95eed0561eeb17'
ANNOTATIONS = {
    'train/labelv2.txt': 'ddb6f7c92f2e5d0f0eabc80cb3407bde5d3f8f19',
    'val/labelv2.txt': '1c9a73173523640b67be4767272290f92bed3a4e',
    'val/gt/wider_easy_val.mat': '5b78df7c006982ab1a2038149f1f5168b2237f07',
    'val/gt/wider_face_val.mat': '7ac47e55b72d04c5fa237321509b2d2319f8ded2',
    'val/gt/wider_hard_val.mat': '061fcab0761d4fb6a0574756ae5c3d4591aecfc7',
    'val/gt/wider_medium_val.mat': '9670c7a96d8af29c59d6b08cab7765b8bc427182',
}


class RemoteZip(io.RawIOBase):
    def __init__(self, url, budget=None):
        self.url = url
        self.budget = budget
        self.cached_start = 0
        self.cached = b''
        with urlopen(Request(url, method='HEAD'), timeout=30) as response:
            self.size = int(response.headers['Content-Length'])
            self.url = response.geturl()
        address = urlsplit(self.url)
        self.connection = http.client.HTTPSConnection(address.netloc, timeout=60)
        self.request_path = address.path + ('?' + address.query if address.query else '')
        self.position = 0
        self.downloaded = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def prefetch(self, start, size):
        size = min(size, self.size - start)
        if size > 8 * 1024 * 1024:
            raise RuntimeError('Requisicao excessiva; download parcial interrompido.')
        if self.budget is not None:
            self.budget.reserve(size)
        self.connection.request('GET', self.request_path, headers={'Range': f'bytes={start}-{start + size - 1}'})
        with self.connection.getresponse() as response:
            if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                raise RuntimeError('Servidor nao respeitou a leitura parcial; download interrompido.')
            payload = response.read(size + 1)
        if len(payload) != size:
            raise RuntimeError('Resposta parcial com tamanho incorreto.')
        self.cached_start = start
        self.cached = payload
        self.downloaded += size

    def close(self):
        if hasattr(self, 'connection'):
            self.connection.close()
        super().close()

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size, self.size - self.position)
        if size <= 0:
            return b''
        start = self.position
        if not (self.cached_start <= start and start + size <= self.cached_start + len(self.cached)):
            self.prefetch(start, size)
        offset = start - self.cached_start
        self.position += size
        return self.cached[offset:offset + size]


class DownloadBudget:
    def __init__(self, path):
        self.path = path
        self.used = json.loads(path.read_text())['reserved_bytes'] if path.exists() else 0

    def reserve(self, size):
        if self.used + size > 300_000_000:
            raise RuntimeError('Limite autorizado de 300 MB atingido; download interrompido.')
        self.used += size
        self.path.write_text(json.dumps({'reserved_bytes': self.used}) + '\n')


def select_records(records, count, seed):
    buckets = defaultdict(list)
    for record in records:
        boxes = [ann.bbox for ann in record.annotations]
        if not boxes or any(not all(math.isfinite(float(x)) for x in box) or
            not (0 <= box[0] < box[2] <= record.width and 0 <= box[1] < box[3] <= record.height) for box in boxes):
            continue
        scale = statistics.median(math.sqrt((box[2] - box[0]) * (box[3] - box[1])) / max(record.width, record.height) for box in boxes)
        size_bin = 0 if scale < 0.03 else 1 if scale < 0.1 else 2
        crowd_bin = 0 if len(boxes) < 5 else 1 if len(boxes) < 20 else 2
        buckets[(record.filename.split('/')[0], size_bin, crowd_bin)].append(record)
    rng = random.Random(seed)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        available = [key for key in keys if buckets[key]]
        if not available:
            raise RuntimeError('Imagens validas insuficientes para a selecao.')
        for key in available:
            selected.append(buckets[key].pop())
            if len(selected) == count:
                break
    return sorted(selected, key=lambda record: record.filename)


def write_subset_annotations(source, destination, names):
    blocks = {}
    current = None
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith('#'):
            current = line[1:].split()[0]
            blocks[current] = []
        if current is not None:
            blocks[current].append(line)
    payload = ''.join(''.join(blocks[name]) for name in names)
    if destination.exists() and destination.read_text() != payload:
        raise RuntimeError('Anotacoes do subconjunto existentes divergem.')
    destination.write_text(payload)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=root / 'data/widerface')
    parser.add_argument('--train-images', type=int, default=16)
    parser.add_argument('--val-images', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if not 8 <= args.train_images <= 1000 or not 1 <= args.val_images <= 200:
        parser.error('Limite: 1000 imagens de treino e 200 de validacao.')
    if args.seed is None and (args.train_images > 16 or args.val_images > 16):
        parser.error('Uma selecao ampliada exige --seed.')
    initial_size = sum(p.stat().st_size for p in args.root.rglob('*') if p.is_file())
    budget = DownloadBudget(args.root / 'finetune_transfer.json') if args.seed is not None else None
    for relative, expected in ANNOTATIONS.items():
        destination = args.root / 'labelv2' / relative
        if destination.exists():
            payload = destination.read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/{REVISION}/data/widerface/labelv2/{relative}'
            with urlopen(url, timeout=60) as response:
                payload = response.read()
        blob = b'blob ' + str(len(payload)).encode() + b'\0' + payload
        if hashlib.sha1(blob).hexdigest() != expected:
            raise RuntimeError(f'Anotacao oficial divergente: {relative}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(payload)
    manifest = {'upstream_revision': REVISION, 'data_revision': DATA_REVISION, 'splits': {}}
    for split, count in [('train', args.train_images), ('val', args.val_images)]:
        records = parse_labelv2_file(args.root / 'labelv2' / split / 'labelv2.txt', test_mode=split == 'val')
        records = select_records(records, count, args.seed + (split == 'val')) if args.seed is not None else records[:count]
        prefix = f'WIDER_{split}/images/'
        url = f'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/{DATA_REVISION}/data/WIDER_{split}.zip'
        remote = RemoteZip(url, budget)
        selected = []
        with ZipFile(remote) as archive:
            for record in records:
                name = prefix + record.filename
                destination = args.root / name
                if not destination.resolve().is_relative_to(args.root.resolve()):
                    raise RuntimeError('Caminho de imagem invalido.')
                if destination.exists() and args.seed is not None:
                    payload = destination.read_bytes()
                    info = archive.getinfo(name)
                    if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                        raise RuntimeError(f'Imagem existente diverge do ZIP: {name}')
                else:
                    info = archive.getinfo(name)
                    remote.prefetch(info.header_offset, 30 + len(info.filename.encode()) + len(info.extra) + info.compress_size + 256)
                    payload = archive.read(name)
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or (image.shape[1], image.shape[0]) != (record.width, record.height):
                    raise RuntimeError(f'Imagem ou dimensoes invalidas: {name}')
                if sum(p.stat().st_size for p in args.root.rglob('*') if p.is_file()) - initial_size + len(payload) > 1_000_000_000:
                    raise RuntimeError('Limite de 1 GB adicional em disco atingido.')
                if destination.exists():
                    if destination.read_bytes() != payload:
                        raise RuntimeError(f'Imagem existente diverge: {name}')
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                selected.append({'filename': record.filename, 'sha256': hashlib.sha256(payload).hexdigest(), 'annotations': len(record.annotations)})
                if len(selected) % 25 == 0 or len(selected) == count:
                    print(f'{split}: {len(selected)}/{count}; transferencia reservada: {budget.used if budget else remote.downloaded} bytes', flush=True)
        if args.seed is not None:
            write_subset_annotations(args.root / 'labelv2' / split / 'labelv2.txt',
                args.root / 'labelv2' / split / f'finetune_seed{args.seed}.txt', [record.filename for record in records])
        manifest['splits'][split] = {'archive': url, 'downloaded_bytes': remote.downloaded, 'images': selected}
    if args.seed is not None:
        manifest['seed'] = args.seed
    output = args.root / (f'finetune_seed{args.seed}.json' if args.seed is not None else 'subset.json')
    payload = json.dumps(manifest, indent=2) + '\n'
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get('seed') != args.seed or any(previous['splits'][split]['images'] != manifest['splits'][split]['images'] for split in ['train', 'val']):
            raise RuntimeError('Manifesto existente diverge; nao sera sobrescrito.')
    else:
        output.write_text(payload)
    print(f'Subconjunto real preparado: {output}')


if __name__ == '__main__':
    main()
