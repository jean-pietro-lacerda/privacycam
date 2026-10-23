import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from yunet_train.cli.export_onnx import export_onnx
from yunet_train.engine.checkpoint import load_checkpoint
from yunet_train.tasks.face import build_yunet, YuNetPostprocessor


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=root / 'checkpoints/yunet_n.pth')
    parser.add_argument('--image', type=Path, default=root.parent / 'test_inputs/multiplos_rostos.jpeg')
    parser.add_argument('--output', type=Path, default=root / 'exports/yunet_n_640.onnx')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('A saida ONNX ja existe; utilize outro caminho.')
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f'Imagem invalida: {args.image}')
    model = build_yunet('yunet_n')
    load_checkpoint(args.checkpoint, model=model, map_location='cpu')
    model.to(args.device).eval()
    scale = 640 / max(image.shape[:2])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    padded = np.zeros((640, 640, 3), dtype=np.float32)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        predictions = model(tensor)
        if not all(torch.isfinite(value).all().item() for group in predictions for value in group):
            raise RuntimeError('Saidas PyTorch nao finitas.')
        detections = YuNetPostprocessor(score_threshold=0.49, nms_threshold=0.3)(predictions)[0]
    print(f'Python {__import__("sys").version.split()[0]}; PyTorch {torch.__version__}; dispositivo {args.device}')
    print(f'Checkpoint carregado estritamente; inferencia Python: {len(detections.scores)} rostos')
    export_onnx(argparse.Namespace(
        checkpoint=args.checkpoint, variant='yunet_n', output_file=args.output,
        shape=[640, 640], opset_version=11, dynamic_export=False, verify=True, device='cpu'
    ))
    onnx.checker.check_model(onnx.load(str(args.output)))
    print(f'ONNX exportado, validado e comparado numericamente: {args.output}')


if __name__ == '__main__':
    main()
