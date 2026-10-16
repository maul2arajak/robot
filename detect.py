# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import argparse
import datetime
import json
import os
import pathlib

# --- IMPORT UNTUK KOMUNIKASI SOCKET ---
import socket
import sys
from pathlib import Path

import torch

# Fix Path untuk Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    cv2,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "0",  # Default ke webcam
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=True,  # Langsung aktifkan view
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    source = str(source)
    not nosave and not source.endswith(".txt")
    webcam = (
        source.isnumeric()
        or source.endswith(".streams")
        or (
            source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
            and not Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        )
    )

    # --- KONFIGURASI SOCKET ---
    SUBSCRIBER_IP = "192.168.137.1"
    PORT = 9999
    msg_id = 1
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect((SUBSCRIBER_IP, PORT))
        LOGGER.info(f"Connected to server {SUBSCRIBER_IP}")
    except Exception as e:
        LOGGER.error(f"Socket Error: {e}")
        sock = None

    # Load Model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, dt = 0, (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred = model(im, augment=augment, visualize=False)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                _p, im0 = path[i], im0s[i].copy()
            else:
                _p, im0 = path, im0s.copy()

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            detections_to_send = []

            if len(det):
                # Rescale koordinat ke ukuran gambar asli (im0)
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Loop setiap objek
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c]

                    # --- HITUNG KOORDINAT ---
                    # xywh: [x_center, y_center, width, height]
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()

                    x_pusat = round(xywh[0], 1)
                    y_pusat = round(xywh[1], 1)
                    lebar = round(xywh[2], 1)
                    tinggi = round(xywh[3], 1)
                    luas = lebar * tinggi

                    # Simpan data objek
                    detections_to_send.append(
                        {"object": label, "conf": round(float(conf), 2), "x": x_pusat, "y": y_pusat, "luas": luas}
                    )

                    # Cetak koordinat ke terminal agar anda bisa lihat langsung
                    print(f"Detected: {label} | Pusat: ({x_pusat}, {y_pusat}) | Size: {lebar}x{tinggi}")

                    # Tambah box di gambar
                    label_str = f"{label} {conf:.2f} ({x_pusat},{y_pusat})"
                    annotator.box_label(xyxy, label_str, color=colors(c, True))

            # --- KIRIM DATA KE WINDOWS ---
            if sock:
                payload = {
                    "id": msg_id,
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "data": detections_to_send,
                }
                try:
                    sock.sendall((json.dumps(payload) + "\n").encode())
                    msg_id += 1
                except:
                    sock = None

            # Tampilkan Gambar
            if view_img:
                cv2.imshow("Qiqi Vision", annotator.result())
                cv2.waitKey(1)

    if sock:
        sock.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_false")  # default sudah true di run()
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--line-thickness", default=3, type=int)
    parser.add_argument("--hide-labels", default=False, action="store_true")
    parser.add_argument("--hide-conf", default=False, action="store_true")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
