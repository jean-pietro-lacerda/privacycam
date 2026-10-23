import argparse
import ipaddress
import json
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--client-ip')
    args = parser.parse_args()
    host = ipaddress.IPv4Address(args.host)
    client = ipaddress.IPv4Address(args.client_ip or '127.0.0.1')
    if host.is_unspecified or host.is_multicast or not (host.is_private or host.is_loopback):
        parser.error('Utilize loopback ou o IP exato da interface Windows do WSL.')
    if not host.is_loopback and (not args.client_ip or client.is_unspecified or not client.is_private):
        parser.error('NAT exige --client-ip com o IP exato do WSL.')
    if args.camera < 0 or not 1 <= args.port <= 65535:
        parser.error('Indice de camera ou porta invalido.')
    if os.name != 'nt':
        parser.error('Execute este adaptador no Python do Windows.')
    import cv2
    camera = None
    server = None
    try:
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            candidate = cv2.VideoCapture(args.camera, backend)
            candidate.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            candidate.set(cv2.CAP_PROP_FPS, 15)
            candidate.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = candidate.read() if candidate.isOpened() else (False, None)
            if ok and frame is not None and frame.size:
                camera = candidate
                break
            candidate.release()
        if camera is None:
            raise RuntimeError('A camera Windows nao abriu ou nao produziu frames reais.')

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(3)

            def log_message(self, format, *values):
                pass

            def do_GET(self):
                if self.client_address[0] != str(client):
                    self.send_error(403)
                    return
                if self.path == '/health':
                    payload = json.dumps({
                        'camera_open': camera.isOpened(),
                        'width': int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    }).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path != '/video':
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    while True:
                        started = time.monotonic()
                        ok, image = camera.read()
                        if not ok or image is None or not image.size:
                            raise RuntimeError('Falha de captura da camera Windows.')
                        ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not ok:
                            raise RuntimeError('Falha de codificacao JPEG.')
                        payload = jpeg.tobytes()
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(payload)).encode() + b'\r\n\r\n' + payload + b'\r\n')
                        self.wfile.flush()
                        time.sleep(max(0, 1 / 15 - (time.monotonic() - started)))
                except (ConnectionError, socket.timeout):
                    return

        server = HTTPServer((str(host), args.port), Handler)
        print(f'Camera real: {frame.shape[1]}x{frame.shape[0]}; http://{host}:{args.port}/video; cliente permitido: {client}', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        if camera is not None:
            camera.release()


if __name__ == '__main__':
    main()
