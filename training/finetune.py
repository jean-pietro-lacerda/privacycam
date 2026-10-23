import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from yunet_train.engine.checkpoint import load_checkpoint, load_model_weights_only, save_checkpoint
from yunet_train.tasks.face import (
    WIDERFaceDataset, YuNetCriterion, build_eval_transforms, build_train_transforms,
    build_yunet, collate_face_samples, evaluate_loss, get_train_crop_choice, train_one_epoch,
)


def validate_data(root, seed, image_size):
    manifest = json.loads((root / f'finetune_seed{seed}.json').read_text())
    if manifest['upstream_revision'] != 'dca340aa082c71081a68d17db8e58b33a58a914b' or manifest['seed'] != seed:
        raise RuntimeError('Revisao ou seed do dataset incorreta.')
    datasets = {}
    hashes = {}
    names = {}
    for split, count in [('train', 1000), ('val', 200)]:
        dataset = WIDERFaceDataset(root / 'labelv2' / split / f'finetune_seed{seed}.txt',
            root / f'WIDER_{split}/images', test_mode=split == 'val', transform=build_eval_transforms(image_size))
        official = WIDERFaceDataset(root / 'labelv2' / split / 'labelv2.txt', dataset.img_prefix, test_mode=split == 'val')
        official = {record.filename: record for record in official.records}
        entries = {entry['filename']: entry for entry in manifest['splits'][split]['images']}
        if len(dataset) != count or len(entries) != count:
            raise RuntimeError(f'Quantidade incorreta de imagens: {split}')
        hashes[split] = set()
        names[split] = set()
        annotations = 0
        for index, record in enumerate(dataset.records):
            entry = entries[record.filename]
            path = dataset.img_prefix / record.filename
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if digest != entry['sha256'] or image is None or image.shape[:2] != (record.height, record.width):
                raise RuntimeError(f'Imagem invalida: {path}')
            original = official[record.filename]
            if len(record.annotations) != len(original.annotations):
                raise RuntimeError('Anotacoes divergentes.')
            for ann, expected in zip(record.annotations, original.annotations):
                if not np.array_equal(ann.bbox, expected.bbox) or not np.array_equal(ann.keypoints, expected.keypoints) or ann.ignore != expected.ignore:
                    raise RuntimeError('Anotacao alterada.')
                x1, y1, x2, y2 = ann.bbox
                if not np.isfinite(ann.bbox).all() or not (0 <= x1 < x2 <= record.width and 0 <= y1 < y2 <= record.height):
                    raise RuntimeError('Bounding box invalida.')
            sample = dataset[index]
            if sample.image.shape != (3, image_size, image_size) or not torch.isfinite(sample.image).all():
                raise RuntimeError('Preprocessamento invalido.')
            hashes[split].add(digest)
            names[split].add(record.filename)
            annotations += len(record.annotations)
        categories = len({name.split('/')[0] for name in names[split]})
        print(f'Dataset {split}: {len(dataset)} imagens, {annotations} anotacoes, {categories} categorias; integridade e preprocessamento validos', flush=True)
        datasets[split] = dataset
    if hashes['train'] & hashes['val'] or names['train'] & names['val']:
        raise RuntimeError('Sobreposicao entre treino e validacao.')
    datasets['train'].transform = build_train_transforms(image_size, crop_choice=get_train_crop_choice('yunet_n'))
    return datasets


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    values = state['numpy']
    np.random.set_state((values[0], np.array(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=root / 'data/widerface')
    parser.add_argument('--pretrained', type=Path, default=root / 'checkpoints/yunet_n.pth')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--epochs', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--batch-size', type=int, choices=[1, 2, 4], default=4)
    parser.add_argument('--image-size', type=int, choices=[320, 640], default=640)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (root.parent / '.venv').resolve():
        parser.error('Execute com .venv/bin/python.')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Learning rate invalida.')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    datasets = validate_data(args.dataset, args.seed, args.image_size)
    if args.validate_only:
        return
    if not torch.cuda.is_available():
        parser.error('CUDA indisponivel; treinamento nao sera iniciado em CPU.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    device = torch.device('cuda')
    model = build_yunet('yunet_n').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    criterion = YuNetCriterion(strides=(8, 16, 32))
    prefix = root / 'checkpoints' / f'finetune_seed{args.seed}'
    metrics_path = prefix.with_suffix('.csv')
    latest_path = prefix.with_name(prefix.name + '_latest.pth')
    best_path = prefix.with_name(prefix.name + '_best.pth')
    config = {'variant': 'yunet_n', 'seed': args.seed, 'batch_size': args.batch_size,
        'image_size': args.image_size, 'lr': args.lr, 'upstream_revision': 'dca340aa082c71081a68d17db8e58b33a58a914b',
        'dataset_sha256': hashlib.sha256((args.dataset / f'finetune_seed{args.seed}.json').read_bytes()).hexdigest()}
    start = 1
    best = math.inf
    baseline = None
    if args.resume:
        state = load_checkpoint(args.resume, model=model, optimizer=optimizer)
        for key, value in config.items():
            if state['config'][key] != value:
                raise RuntimeError(f'Configuracao de retomada divergente: {key}')
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        start = state['epoch'] + 1
        best = state['metrics']['best_val_loss']
        baseline = state['config']['baseline_val_loss']
        restore_rng(state['config']['rng'])
        print(f'Retomada carregada: epoca {start}; otimizador e RNG restaurados', flush=True)
    else:
        if latest_path.exists() or best_path.exists() or metrics_path.exists():
            parser.error('Resultados deste fine-tuning ja existem; utilize --resume.')
        load_model_weights_only(args.pretrained, model=model)
        print('Pesos pre-treinados carregados; novo otimizador SGD', flush=True)
    train_loader = DataLoader(
        datasets['train'], batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_face_samples
    )
    val_loader = DataLoader(
        datasets['val'], batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_face_samples
    )

    def checked_loss(predictions, **targets):
        losses = criterion(predictions, **targets)
        if not all(value.is_cuda and torch.isfinite(value).all().item() for value in losses.values()):
            raise RuntimeError('Loss nao finita ou fora da GPU.')
        return losses

    gradient_steps = 0

    def check_gradients(optimizer, positional, named):
        nonlocal gradient_steps
        gradients = [param.grad for param in model.parameters() if param.grad is not None]
        if not gradients or not all(value.is_cuda and torch.isfinite(value).all().item() for value in gradients):
            raise RuntimeError('Gradientes invalidos.')
        if not any(torch.count_nonzero(value).item() for value in gradients):
            raise RuntimeError('Gradientes nulos.')
        gradient_steps += 1

    def save(path, epoch, metrics):
        temporary = path.with_suffix('.tmp')
        save_checkpoint(path=temporary, model=model, optimizer=optimizer, epoch=epoch,
            config={**config, 'baseline_val_loss': baseline, 'rng': rng_state()}, metrics=metrics)
        temporary.replace(path)

    if start > args.epochs:
        print('Execucao ja concluida; nenhum treinamento adicional.', flush=True)
        return
    if baseline is None:
        baseline = evaluate_loss(model=model, criterion=checked_loss, data_loader=val_loader, device=device).loss
        if not math.isfinite(baseline):
            raise RuntimeError('Loss inicial de validacao nao finita.')
        with metrics_path.open('x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss'])
            writer.writerow([0, '', baseline, ''])
        print(f'Validacao pre-treinada: {baseline:.6f}', flush=True)
    hook = optimizer.register_step_pre_hook(check_gradients)
    try:
        for epoch in range(start, args.epochs + 1):
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            gradient_steps = 0
            train = train_one_epoch(model=model, criterion=checked_loss, data_loader=train_loader,
                optimizer=optimizer, device=device, epoch=epoch, log_interval=50, logger=lambda text: print(text, flush=True))
            if gradient_steps != len(train_loader) or not math.isfinite(train.loss):
                raise RuntimeError('Passos de treinamento invalidos.')
            if not any(not torch.equal(before[name], value) for name, value in model.named_parameters()):
                raise RuntimeError('Pesos nao atualizados.')
            val = evaluate_loss(model=model, criterion=checked_loss, data_loader=val_loader, device=device)
            if not math.isfinite(val.loss):
                raise RuntimeError('Loss de validacao nao finita.')
            improved = val.loss < best
            best = min(best, val.loss)
            metrics = {'train_loss': train.loss, 'val_loss': val.loss, 'best_val_loss': best}
            epoch_path = prefix.with_name(prefix.name + f'_epoch{epoch}.pth')
            if epoch_path.exists():
                raise RuntimeError('Checkpoint periodico ja existe.')
            save(epoch_path, epoch, metrics)
            save(latest_path, epoch, metrics)
            if improved:
                save(best_path, epoch, metrics)
            reloaded = build_yunet('yunet_n')
            load_checkpoint(epoch_path, model=reloaded)
            if not all(torch.equal(value.cpu(), reloaded.state_dict()[name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Checkpoint recarregado diverge.')
            with metrics_path.open('a', newline='') as file:
                csv.writer(file).writerow([epoch, train.loss, val.loss, best])
            print(f'Epoca {epoch}: treino={train.loss:.6f}; validacao={val.loss:.6f}; melhor={best:.6f}; gradientes validos={gradient_steps}', flush=True)
    finally:
        hook.remove()
    print(f'GPU: {torch.cuda.get_device_name()}; pico alocado={torch.cuda.max_memory_allocated()/1024**2:.2f} MiB; reservado={torch.cuda.max_memory_reserved()/1024**2:.2f} MiB', flush=True)
    print(f'Checkpoint selecionado: {best_path}; validacao inicial={baseline:.6f}; melhor treinada={best:.6f}', flush=True)


if __name__ == '__main__':
    main()
