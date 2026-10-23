import argparse
from dataclasses import asdict
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from yunet_train.engine.checkpoint import load_checkpoint, load_model_weights_only, save_checkpoint
from yunet_train.tasks.face import (
    build_yunet, build_train_transforms, collate_face_samples,
    get_train_crop_choice, train_one_epoch, WIDERFaceDataset, YuNetCriterion,
)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=root / 'data/widerface')
    parser.add_argument('--checkpoint', type=Path, default=root / 'checkpoints/yunet_n.pth')
    parser.add_argument('--output', type=Path, default=root / 'checkpoints/smoke.pth')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--samples', type=int, choices=range(8, 17), default=16)
    parser.add_argument('--batch-size', type=int, choices=[1, 2], default=2)
    parser.add_argument('--image-size', type=int, choices=[320, 640], default=320)
    parser.add_argument('--ann-file', type=Path)
    args = parser.parse_args()
    annotations = args.ann_file or args.dataset / 'labelv2/train/labelv2.txt'
    images = args.dataset / 'WIDER_train/images'
    missing = [str(path) for path in [annotations, images] if not path.exists()]
    if missing:
        parser.exit(2, 'Treinamento bloqueado; faltam: ' + ', '.join(missing) + '\n')
    if args.output.exists():
        parser.error('Checkpoint de saida ja existe.')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA indisponivel; o teste CUDA nao sera substituido por CPU.')
        torch.cuda.reset_peak_memory_stats()
    dataset = WIDERFaceDataset(ann_file=annotations, img_prefix=images,
        transform=build_train_transforms(image_size=args.image_size, crop_choice=get_train_crop_choice('yunet_n')))
    dataset.records = dataset.records[:args.samples]
    if not dataset.records or not dataset.records[0].annotations:
        parser.error('Necessaria pelo menos uma imagem com anotacoes reais.')
    if len(dataset) != args.samples:
        parser.error('Quantidade de imagens anotadas insuficiente.')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_face_samples)
    model = build_yunet('yunet_n')
    load_model_weights_only(args.checkpoint, model=model)
    model.to(args.device)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    losses = []
    gradient_steps = []
    criterion = YuNetCriterion(strides=(8, 16, 32))

    def checked_loss(predictions, **targets):
        values = criterion(predictions, **targets)
        if not all(value.device.type == args.device and torch.isfinite(value).all().item() for value in values.values()):
            raise RuntimeError('Loss nao finita ou calculada no dispositivo incorreto.')
        losses.append(sum(value.detach().item() for value in values.values()))
        return values

    def check_gradients(optimizer, positional, named):
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        if not gradients or not all(torch.isfinite(value).all().item() for value in gradients):
            raise RuntimeError('Gradientes ausentes ou nao finitos.')
        if not any(torch.count_nonzero(value).item() for value in gradients):
            raise RuntimeError('Todos os gradientes sao nulos.')
        gradient_steps.append(len(gradients))

    hook = optimizer.register_step_pre_hook(check_gradients)
    try:
        stats = train_one_epoch(model=model, criterion=checked_loss,
            data_loader=loader, optimizer=optimizer, device=args.device, epoch=1,
            log_interval=1, logger=print)
    finally:
        hook.remove()
    if stats.steps != len(loader) or len(gradient_steps) != stats.steps or not math.isfinite(stats.loss):
        raise RuntimeError('Passo de treinamento invalido.')
    if not any(not torch.equal(before[name], value) for name, value in model.named_parameters()):
        raise RuntimeError('Os pesos nao foram atualizados.')
    save_checkpoint(path=args.output, model=model, optimizer=optimizer, epoch=1,
        config={'variant': 'yunet_n', 'technical_smoke_test': True, 'samples': args.samples,
            'batch_size': args.batch_size, 'image_size': args.image_size, 'device': args.device,
            'upstream_revision': 'dca340aa082c71081a68d17db8e58b33a58a914b'},
        metrics={**asdict(stats), 'first_batch_loss': losses[0], 'last_batch_loss': losses[-1]})
    reloaded = build_yunet('yunet_n')
    load_checkpoint(args.output, model=reloaded)
    if not all(torch.equal(value.cpu(), reloaded.state_dict()[name]) for name, value in model.state_dict().items()):
        raise RuntimeError('Checkpoint recarregado diverge.')
    print(f'Forward, loss, backward, atualizacao e recarga: sucesso; loss={stats.loss}; checkpoint={args.output}')
    print(f'Dispositivo: {next(model.parameters()).device}; passos com gradientes validos: {len(gradient_steps)}')
    print(f'Loss primeiro/ultimo minibatch: {losses[0]} / {losses[-1]}')
    if args.device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}; pico alocado: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB; pico reservado: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
