#!/usr/bin/env python3
import argparse
import collections
import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from keras.models import load_model
    from lib.resnet_model import Resnet3DBuilder
    KERAS_AVAILABLE = True
except Exception:
    load_model = None
    Resnet3DBuilder = None
    KERAS_AVAILABLE = False

try:
    from lib.data_loader import DataLoader
    DATALOADER_AVAILABLE = True
except Exception:
    DataLoader = None
    DATALOADER_AVAILABLE = False

try:
    import mss
except Exception as exc:
    mss = None
    print("WARNING: mss not available; screen capture will be disabled.", file=sys.stderr)

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    print("WARNING: mediapipe not available; pinch detection will be disabled.", file=sys.stderr)

try:
    import pyautogui
except Exception:
    pyautogui = None
    print("WARNING: pyautogui not available; auto-scroll will be disabled.", file=sys.stderr)

try:
    import pytesseract
except Exception:
    pytesseract = None
    print("WARNING: pytesseract not available; OCR will be disabled.", file=sys.stderr)

try:
    import pygetwindow as gw
except Exception:
    gw = None


@dataclass
class RuntimeConfig:
    model_path: Optional[str]
    frame_model: Optional[str]
    frame_labels_json: Optional[str]
    frame_input: int
    palm_frames: int
    palm_spread: float
    pinch_dist_thresh: float
    drag_gain: float
    edge_extrapolate_thresh_px: int
    edge_extrapolate_step_px: int
    cam_norm_left: float
    cam_norm_right: float
    cam_norm_top: float
    cam_norm_bottom: float
    video_output_dir: str
    double_palm_window_s: float
    frames: int
    height: int
    width: int
    labels_path: str
    confidence_threshold: float
    smooth_window: int
    output_dir: str
    show_overlay: bool
    preview: bool


class GestureClassifier:
    def __init__(self, config: RuntimeConfig, labels: List[str]):
        self.config = config
        self.labels = labels
        self.num_classes = len(labels)
        self.model = self._load_model()
        self.frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=config.frames)
        self.pred_smooth: Deque[int] = collections.deque(maxlen=config.smooth_window)

    def _build_architecture(self):
        if not KERAS_AVAILABLE:
            raise RuntimeError("Keras/TensorFlow not available to build architecture.")
        input_shape = (self.config.frames, self.config.height, self.config.width, 3)
        return Resnet3DBuilder.build_resnet_101(
            input_shape=input_shape,
            num_outputs=self.num_classes,
            reg_factor=1e-4,
            drop_rate=0.0,
        )

    def _load_model(self):
        if self.config.model_path and os.path.exists(self.config.model_path) and KERAS_AVAILABLE:
            try:
                return load_model(self.config.model_path)
            except Exception:
                model = self._build_architecture()
                model.load_weights(self.config.model_path)
                return model
        print("INFO: Running without a trained model; gesture classification disabled.", file=sys.stderr)
        return None

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = frame_bgr[y0:y0 + side, x0:x0 + side]
        resized = cv2.resize(crop, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        return arr

    def predict_label(self) -> Tuple[str, float]:
        if self.model is None:
            return "No gesture", 0.0
        if len(self.frame_buffer) < self.config.frames:
            return "No gesture", 0.0
        x = np.stack(list(self.frame_buffer), axis=0)
        x = np.expand_dims(x, axis=0)
        preds = self.model.predict(x, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])
        self.pred_smooth.append(class_idx)
        vote_idx = max(set(self.pred_smooth), key=self.pred_smooth.count)
        return self.labels[vote_idx], confidence

    def push_frame(self, frame_bgr: np.ndarray):
        self.frame_buffer.append(self.preprocess_frame(frame_bgr))


class FrameGestureClassifier:
    def __init__(self, model_path: str, labels: List[str], input_size: int = 128):
        if not KERAS_AVAILABLE:
            raise RuntimeError("Keras/TensorFlow not available.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = load_model(model_path)
        self.labels = labels
        self.input_size = input_size
        self.pred_smooth: Deque[int] = collections.deque(maxlen=5)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = frame_bgr[y0:y0 + side, x0:x0 + side]
        resized = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict_label(self, frame_bgr: np.ndarray) -> Tuple[str, float]:
        x = self.preprocess(frame_bgr)
        preds = self.model.predict(x, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        conf = float(preds[class_idx])
        self.pred_smooth.append(class_idx)
        vote_idx = max(set(self.pred_smooth), key=self.pred_smooth.count)
        return self.labels[vote_idx], conf


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def capture_region(bbox_xyxy: Tuple[int, int, int, int], output_dir: str) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width == 0 or height == 0:
        raise ValueError("Empty selection.")
    ensure_output_dir(output_dir)
    out_path = os.path.join(output_dir, f"{now_timestamp()}.png")
    if pyautogui is not None:
        full = pyautogui.screenshot()
        crop_box = (int(left), int(top), int(left + width), int(top + height))
        full.crop(crop_box).save(out_path)
    else:
        if mss is None:
            raise RuntimeError("Neither pyautogui nor mss available; cannot capture screen.")
        with mss.mss() as sct:
            monitor = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
            im = sct.grab(monitor)
            img = Image.frombytes("RGB", im.size, im.rgb)
            img.save(out_path)
    return out_path


def stitch_vertical(images: List[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("No images to stitch.")
    widths = [im.width for im in images]
    target_w = max(widths)
    resized = [im if im.width == target_w else im.resize((target_w, int(im.height * (target_w / im.width))), Image.BILINEAR) for im in images]
    total_h = sum(im.height for im in resized)
    out = Image.new("RGB", (target_w, total_h))
    y = 0
    for im in resized:
        out.paste(im, (0, y))
        y += im.height
    return out


class LongScreenshotWorker(threading.Thread):
    def __init__(self, bbox_xyxy: Tuple[int, int, int, int], output_dir: str, scroll_step: int = 600, interval_s: float = 0.8):
        super(LongScreenshotWorker, self).__init__(daemon=True)
        self.bbox = bbox_xyxy
        self.output_dir = output_dir
        self.scroll_step = scroll_step
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.captures: List[Image.Image] = []
        self.out_path: Optional[str] = None

    def run(self):
        if mss is None:
            return
        ensure_output_dir(self.output_dir)
        with mss.mss() as sct:
            while not self.stop_event.is_set():
                try:
                    x1, y1, x2, y2 = self.bbox
                    left = min(x1, x2)
                    top = min(y1, y2)
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    monitor = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
                    im = sct.grab(monitor)
                    img = Image.frombytes("RGB", im.size, im.rgb)
                    self.captures.append(img)
                    if pyautogui is not None:
                        pyautogui.scroll(-self.scroll_step)
                    time.sleep(self.interval_s)
                except Exception:
                    break
        if self.captures:
            stitched = stitch_vertical(self.captures)
            self.out_path = os.path.join(self.output_dir, f"longsnip_{now_timestamp()}.png")
            stitched.save(self.out_path)

    def stop(self):
        self.stop_event.set()


class PinchTracker:
    def __init__(self, pinch_dist_thresh: float = 0.07, palm_spread_thresh: float = 0.20):
        self.available = mp is not None
        self.pinch_dist_thresh = float(pinch_dist_thresh)
        self.palm_spread_thresh = float(palm_spread_thresh)
        if self.available:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.screen_w = int(cv2.getWindowImageRect("HandSnip")[2]) if "HandSnip" in [w for w in list(map(lambda x: x, []))] else None
        self.prev_pinch = False

    def close(self):
        if self.available and self.hands:
            self.hands.close()

    def detect(self, frame_bgr: np.ndarray, screen_size: Tuple[int, int]) -> Tuple[bool, Optional[Tuple[int, int]], bool, bool, bool, bool, bool, Tuple[float, float]]:
        if not self.available:
            return False, None, False, False, False, False, False, (0.5, 0.5)
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            self.prev_pinch = False
            return False, None, False, False, False, False, False, (0.5, 0.5)
        hand_landmarks = results.multi_hand_landmarks[0]
        lx = hand_landmarks.landmark[8].x
        ly = hand_landmarks.landmark[8].y
        tx = hand_landmarks.landmark[4].x
        ty = hand_landmarks.landmark[4].y
        dist = np.hypot(lx - tx, ly - ty)
        pinch = dist < self.pinch_dist_thresh
        wrist = hand_landmarks.landmark[0]
        idx_tip = hand_landmarks.landmark[8]
        mid_tip = hand_landmarks.landmark[12]
        rng_tip = hand_landmarks.landmark[16]
        pky_tip = hand_landmarks.landmark[20]
        d_idx = np.hypot(idx_tip.x - wrist.x, idx_tip.y - wrist.y)
        d_mid = np.hypot(mid_tip.x - wrist.x, mid_tip.y - wrist.y)
        d_rng = np.hypot(rng_tip.x - wrist.x, rng_tip.y - wrist.y)
        d_pky = np.hypot(pky_tip.x - wrist.x, pky_tip.y - wrist.y)
        spread = abs(idx_tip.x - pky_tip.x)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_dist = np.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
        open_palm = (
            (not pinch)
            and (d_idx > 0.22 and d_mid > 0.24 and d_rng > 0.22 and d_pky > 0.20)
            and (thumb_dist > 0.18)
            and (spread > self.palm_spread_thresh)
        )
        fist = (not pinch) and (spread < self.palm_spread_thresh * 0.5) and (d_idx < 0.16 and d_mid < 0.18 and d_rng < 0.16 and d_pky < 0.14)
        thumb_dist = np.hypot(hand_landmarks.landmark[4].x - wrist.x, hand_landmarks.landmark[4].y - wrist.y)
        others_curled = (d_idx < 0.18 and d_mid < 0.18 and d_rng < 0.18 and d_pky < 0.18)
        dy_thumb = hand_landmarks.landmark[4].y - wrist.y
        thumbs_up = (not pinch) and others_curled and (thumb_dist > 0.22) and (dy_thumb < -0.05)
        thumbs_down = (not pinch) and others_curled and (thumb_dist > 0.22) and (dy_thumb > 0.05)
        
        thumb_tip = hand_landmarks.landmark[4]
        idx_tip = hand_landmarks.landmark[8]
        
        thumb_idx_dist = np.hypot(thumb_tip.x - idx_tip.x, thumb_tip.y - idx_tip.y)
        
        circle = (thumb_idx_dist < 0.12 and not fist and not open_palm)
        
        screen_w, screen_h = screen_size
        index_px = (int(lx * screen_w), int(ly * screen_h))
        self.prev_pinch = pinch
        return pinch, index_px, open_palm, fist, thumbs_up, thumbs_down, circle, (lx, ly)


class HandSnipApp:
    def __init__(self, config: RuntimeConfig, labels: List[str]):
        self.config = config
        self.labels = labels
        self.classifier = GestureClassifier(config, labels)
        self.cap = self._open_camera()
        self.state = "IDLE"
        self.anchor_xy: Optional[Tuple[int, int]] = None
        self.current_xy: Optional[Tuple[int, int]] = None
        self.last_snip_path: Optional[str] = None
        self.last_action_ts = time.time()
        self.status_text = "IDLE"
        self.pinch = PinchTracker(pinch_dist_thresh=self.config.pinch_dist_thresh, palm_spread_thresh=self.config.palm_spread)
        self.long_worker: Optional[LongScreenshotWorker] = None
        self.rec_worker: Optional[ScreenRecorderWorker] = None
        self.last_open_palm_ts: float = 0.0
        if mss is not None:
            try:
                with mss.mss() as sct:
                    mons = getattr(sct, "monitors", [])
                    if isinstance(mons, list) and len(mons) >= 2:
                        mon = mons[1]
                    elif isinstance(mons, list) and len(mons) == 1:
                        mon = mons[0]
                    else:
                        mon = None
                    if mon:
                        self.screen_size = (mon.get("width", 1920), mon.get("height", 1080))
                    else:
                        if pyautogui is not None:
                            sz = pyautogui.size()
                            self.screen_size = (sz.width, sz.height)
                        else:
                            self.screen_size = (1920, 1080)
            except Exception:
                if pyautogui is not None:
                    sz = pyautogui.size()
                    self.screen_size = (sz.width, sz.height)
                else:
                    self.screen_size = (1920, 1080)
        else:
            self.screen_size = (1920, 1080)
        self.frame_classifier: Optional[FrameGestureClassifier] = None
        self.frozen_pil: Optional[Image.Image] = None
        self.frozen_bgr: Optional[np.ndarray] = None
        self.freeze_window_name = "HandSnip Freeze"
        self.freeze_window_open = False
        self._last_freeze_ts: float = 0.0
        self._open_palm_streak: int = 0
        self.selection_finalized: bool = False
        self.last_index_xy: Optional[Tuple[int, int]] = None
        self.smoothed_index_xy: Optional[Tuple[float, float]] = None
        self._pinch_prev: bool = False
        self._circle_prev: bool = False
        self.drag_start_screen: Optional[Tuple[int, int]] = None
        self.drag_start_norm: Optional[Tuple[float, float]] = None

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.pinch.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    def _selected_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        if not (self.anchor_xy and self.current_xy):
            return None
        return (self.anchor_xy[0], self.anchor_xy[1], self.current_xy[0], self.current_xy[1])

    def _draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        overlay = frame_bgr.copy()
        cv2.putText(overlay, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
        cv2.putText(overlay, self.status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        if self.anchor_xy and self.current_xy:
            x1, y1 = self.anchor_xy
            x2, y2 = self.current_xy
            sw, sh = self.screen_size
            ph, pw = frame_bgr.shape[:2]
            px1, py1 = int(x1 / sw * pw), int(y1 / sh * ph)
            px2, py2 = int(x2 / sw * pw), int(y2 / sh * ph)
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 255, 0), 2)
        return overlay

    def _enter_freeze(self):
        try:
            if self.freeze_window_open:
                return
            if time.time() - self._last_freeze_ts < 1.0:
                return
            if pyautogui is not None:
                shot = pyautogui.screenshot()
                self.frozen_pil = shot.convert("RGB")
                self.frozen_bgr = cv2.cvtColor(np.array(self.frozen_pil), cv2.COLOR_RGB2BGR)
            elif mss is not None:
                with mss.mss() as sct:
                    mon = sct.monitors[1]
                    im = sct.grab(mon)
                    self.frozen_pil = Image.frombytes("RGB", im.size, im.rgb)
                    self.frozen_bgr = cv2.cvtColor(np.array(self.frozen_pil), cv2.COLOR_RGB2BGR)
            else:
                self.status_text = "No screen capture backend available"
                return
            cv2.namedWindow(self.freeze_window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.freeze_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            if self.frozen_bgr is not None:
                h, w = self.frozen_bgr.shape[:2]
                try:
                    cv2.resizeWindow(self.freeze_window_name, w, h)
                    cv2.moveWindow(self.freeze_window_name, 0, 0)
                except Exception:
                    pass
            self.freeze_window_open = True
            self.selection_finalized = False
            self._transition("SELECTING", "Frozen. Pinch to draw, Circle/Thumb Up to confirm, Thumb Down to cancel")
            self._last_freeze_ts = time.time()
        except Exception as exc:
            self.status_text = f"Freeze failed: {exc}"

    def _exit_freeze(self):
        if self.freeze_window_open:
            try:
                cv2.destroyWindow(self.freeze_window_name)
            except Exception:
                pass
            self.freeze_window_open = False
        self.frozen_pil = None
        self.frozen_bgr = None
        self.selection_finalized = False
    
    def _open_camera(self):
        cap = None
        try_order = [
            (0, cv2.CAP_AVFOUNDATION),
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_AVFOUNDATION),
            (1, cv2.CAP_ANY),
            (2, cv2.CAP_AVFOUNDATION),
            (2, cv2.CAP_ANY),
        ]
        for idx, backend in try_order:
            try:
                tmp = cv2.VideoCapture(idx, backend)
                if tmp and tmp.isOpened():
                    cap = tmp
                    break
                if tmp:
                    tmp.release()
            except Exception:
                continue
        if cap is None:
            cap = cv2.VideoCapture(0)
        return cap

    def _confirm_capture(self) -> Optional[str]:
        if not (self.anchor_xy and self.current_xy):
            self._transition("IDLE", "No region selected")
            self._exit_freeze()
            return None
        try:
            if self.frozen_pil is not None:
                x1, y1, x2, y2 = self.anchor_xy[0], self.anchor_xy[1], self.current_xy[0], self.current_xy[1]
                left = min(x1, x2); top = min(y1, y2); right = max(x1, x2); bottom = max(y1, y2)
                crop = self.frozen_pil.crop((int(left), int(top), int(right), int(bottom)))
                ensure_output_dir(self.config.output_dir)
                out_path = os.path.join(self.config.output_dir, f"{now_timestamp()}.png")
                crop.save(out_path)
            else:
                out_path = capture_region((self.anchor_xy[0], self.anchor_xy[1], self.current_xy[0], self.current_xy[1]), self.config.output_dir)
            self.last_snip_path = out_path
            self._transition("IDLE", f"Snipped: {out_path}")
        except Exception as exc:
            self._transition("IDLE", f"Capture failed: {exc}")
            out_path = None
        self.anchor_xy, self.current_xy = None, None
        self._exit_freeze()
        return out_path

    def _transition(self, new_state: str, message: str):
        self.state = new_state
        self.status_text = message
        self.last_action_ts = time.time()

    def _handle_gesture(self, label: str, confidence: float):
        normalized = label.strip().lower()
        is_arm = normalized in ("stop sign", "open_palm", "open palm", "open-palm", "open-palm-left", "open-palm-right")
        is_select = normalized in ("sliding two fingers down", "pinch_drag", "pinch-drag", "pinch and drag")
        is_select_stop = normalized in ("sliding two fingers up",)
        is_confirm = normalized in ("thumb up", "circle", "circle-left", "circle-right")
        is_cancel = normalized in ("thumb down",)

        if confidence < self.config.confidence_threshold:
            return
        if self.state == "IDLE" and is_arm:
            self._enter_freeze()
        elif self.state == "SELECTING" and is_confirm:
            self._confirm_capture()
        elif self.state == "SELECTING" and is_cancel:
            self.anchor_xy, self.current_xy = None, None
            self._transition("IDLE", "Cancelled")
            self._exit_freeze()
        if self.state in ("SELECTING", "LONG_CAPTURING") and is_select:
            if self.state != "LONG_CAPTURING":
                bbox = self._selected_bbox()
                if bbox is None:
                    self.status_text = "Select a region first (pinch) for long screenshot"
                else:
                    self.long_worker = LongScreenshotWorker(bbox, self.config.output_dir)
                    self.long_worker.start()
                    self._transition("LONG_CAPTURING", "Long screenshot capturing... Show 'Sliding Two Fingers Up' to stop")
        elif self.state == "LONG_CAPTURING" and is_select_stop:
            if self.long_worker:
                self.long_worker.stop()
                self.long_worker.join(timeout=5.0)
                out_path = self.long_worker.out_path
                self.long_worker = None
                self.anchor_xy, self.current_xy = None, None
                if out_path:
                    self.last_snip_path = out_path
                    self._transition("IDLE", f"Long snip saved: {out_path}")
                else:
                    self._transition("IDLE", "Long snip cancelled or failed")
        if normalized in ("circle", "circle-left", "circle-right"):
            if not self.freeze_window_open and self.state == "IDLE":
                if self.rec_worker:
                    self.rec_worker.stop()
                    self.rec_worker.join(timeout=5.0)
                    end_ts = now_timestamp()
                    if self.rec_worker.out_path and self.rec_worker.start_ts:
                        base_dir = os.path.dirname(self.rec_worker.out_path)
                        new_path = os.path.join(base_dir, f"{self.rec_worker.start_ts}_{end_ts}.mp4")
                        try:
                            os.replace(self.rec_worker.out_path, new_path)
                            self.rec_worker.out_path = new_path
                        except Exception:
                            pass
                    msg = f"Recording saved: {self.rec_worker.out_path}" if self.rec_worker.out_path else "Recording stopped"
                    self.rec_worker = None
                    self._transition(self.state, msg)
                else:
                    ensure_output_dir(self.config.video_output_dir)
                    self.rec_worker = ScreenRecorderWorker(bbox=None, output_dir=self.config.video_output_dir)
                    self.rec_worker.start()
                    self._transition(self.state, "Recording started")
        if normalized in ("swiping right",):
            if pytesseract is None:
                self.status_text = "pytesseract not installed"
            elif not self.last_snip_path or not os.path.exists(self.last_snip_path):
                self.status_text = "No snip available for OCR"
            else:
                try:
                    img = Image.open(self.last_snip_path)
                    text = pytesseract.image_to_string(img)
                    txt_path = os.path.splitext(self.last_snip_path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    self.status_text = f"OCR saved: {txt_path}"
                except Exception as exc:
                    self.status_text = f"OCR failed: {exc}"

    def run(self):
        window_name = "HandSnip"
        print("=== HandSnip Starting ===", file=sys.stderr, flush=True)
        print(f"Preview enabled: {self.config.preview}", file=sys.stderr, flush=True)
        
        try:
            cv2.startWindowThread()
            print("Window thread started successfully", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"WARNING: Failed to start window thread: {e}", file=sys.stderr, flush=True)
        
        if self.config.preview:
            print(f"Creating window '{window_name}'...", file=sys.stderr, flush=True)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            print("Window created", file=sys.stderr, flush=True)
            
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                print("Window properties set", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"WARNING: Could not set window properties: {e}", file=sys.stderr, flush=True)
            
            try:
                cv2.resizeWindow(window_name, 640, 480)
                print("Window resized to 640x480", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"WARNING: Could not resize window: {e}", file=sys.stderr, flush=True)
            
            print(f"Preview window '{window_name}' created. Check if it's visible.", file=sys.stderr, flush=True)
        
        idle_timeout = 8.0
        print("Entering main loop...", file=sys.stderr, flush=True)
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    if self.freeze_window_open and self.frozen_bgr is not None:
                        cv2.imshow(self.freeze_window_name, self.frozen_bgr)
                    if self.config.preview:
                        blank = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, "Waiting for camera frames...", (40, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                        cv2.imshow(window_name, blank)
                        try:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                        except Exception:
                            pass
                    key = cv2.waitKey(10) & 0xFF
                    if key == 27:
                        break
                    continue
                frame = cv2.flip(frame, 1)

                if self.frame_classifier is not None:
                    label, conf = self.frame_classifier.predict_label(frame)
                else:
                    self.classifier.push_frame(frame)
                    label, conf = self.classifier.predict_label()
                self.status_text = f"{label} ({conf:.2f}) | {self.state}"

                self._handle_gesture(label, conf)

                pinch, idx_xy, open_palm, fist, thumbs_up, thumbs_down, circle, norm_xy = self.pinch.detect(frame, self.screen_size)
                if idx_xy:
                    self.last_index_xy = idx_xy
                    if self.smoothed_index_xy is None:
                        self.smoothed_index_xy = (float(idx_xy[0]), float(idx_xy[1]))
                    else:
                        alpha = 0.5
                        sx, sy = self.smoothed_index_xy
                        self.smoothed_index_xy = (alpha * idx_xy[0] + (1 - alpha) * sx,
                                                  alpha * idx_xy[1] + (1 - alpha) * sy)
                if self.state == "IDLE" and not self.freeze_window_open:
                    if open_palm and not pinch:
                        self._open_palm_streak += 1
                        if self._open_palm_streak >= self.config.palm_frames:
                            self._enter_freeze()
                            self._open_palm_streak = 0
                    else:
                        self._open_palm_streak = 0
                if self.state in ("ARMED", "SELECTING"):
                    if pinch and not self._pinch_prev:
                        start_xy = idx_xy or (self.last_index_xy if self.last_index_xy else None)
                        if start_xy and not self.selection_finalized:
                            if self.state == "ARMED" or (self.state == "SELECTING" and self.anchor_xy is None):
                                self.anchor_xy = start_xy
                                self.current_xy = start_xy
                                self.drag_start_screen = start_xy
                                self.drag_start_norm = norm_xy
                                if self.state == "ARMED":
                                    self._transition("SELECTING", "Selecting... Thumbs Up to confirm, Thumbs Down/Fist to cancel")
                    if pinch and not self.selection_finalized:
                        follow = None
                        if self.smoothed_index_xy:
                            follow = (int(self.smoothed_index_xy[0]), int(self.smoothed_index_xy[1]))
                        elif idx_xy:
                            follow = idx_xy
                        elif self.last_index_xy:
                            follow = self.last_index_xy
                        if follow and self.drag_start_norm:
                            sw, sh = self.screen_size
                            nlx, nly = norm_xy
                            fx = (nlx - self.config.cam_norm_left) / max(1e-6, (self.config.cam_norm_right - self.config.cam_norm_left))
                            fy = (nly - self.config.cam_norm_top) / max(1e-6, (self.config.cam_norm_bottom - self.config.cam_norm_top))
                            fx = float(np.clip(fx, 0.0, 1.0))
                            fy = float(np.clip(fy, 0.0, 1.0))
                            sx = (self.drag_start_norm[0] - self.config.cam_norm_left) / max(1e-6, (self.config.cam_norm_right - self.config.cam_norm_left))
                            sy = (self.drag_start_norm[1] - self.config.cam_norm_top) / max(1e-6, (self.config.cam_norm_bottom - self.config.cam_norm_top))
                            sx = float(np.clip(sx, 0.0, 1.0))
                            sy = float(np.clip(sy, 0.0, 1.0))
                            dx = (fx - sx) * sw * max(1.0, float(self.config.drag_gain))
                            dy = (fy - sy) * sh * max(1.0, float(self.config.drag_gain))
                            nx = int(self.anchor_xy[0] + dx)
                            ny = int(self.anchor_xy[1] + dy)
                            if follow[0] > sw - self.config.edge_extrapolate_thresh_px:
                                nx += self.config.edge_extrapolate_step_px
                            if follow[1] > sh - self.config.edge_extrapolate_thresh_px:
                                ny += self.config.edge_extrapolate_step_px
                            if follow[0] < self.config.edge_extrapolate_thresh_px:
                                nx -= self.config.edge_extrapolate_step_px
                            if follow[1] < self.config.edge_extrapolate_thresh_px:
                                ny -= self.config.edge_extrapolate_step_px
                            nx = int(np.clip(nx, 0, sw - 1))
                            ny = int(np.clip(ny, 0, sh - 1))
                            self.current_xy = (nx, ny)
                    else:
                        if (not pinch) and self._pinch_prev and self.anchor_xy and self.current_xy:
                            self.selection_finalized = True
                            self.drag_start_screen = None
                        if (fist or thumbs_down) and self.state == "SELECTING":
                            self.anchor_xy, self.current_xy = None, None
                            self.selection_finalized = False
                            self._transition("IDLE", "Cancelled")
                            self._exit_freeze()
                        if thumbs_up and self.state == "SELECTING" and self.anchor_xy and self.current_xy:
                            self._confirm_capture()
                self._pinch_prev = bool(pinch)
                
                if circle and not self._circle_prev and self.state == "IDLE" and not self.freeze_window_open:
                    if self.rec_worker:
                        self.rec_worker.stop()
                        self.rec_worker.join(timeout=5.0)
                        end_ts = now_timestamp()
                        if self.rec_worker.out_path and self.rec_worker.start_ts:
                            base_dir = os.path.dirname(self.rec_worker.out_path)
                            new_path = os.path.join(base_dir, f"{self.rec_worker.start_ts}_{end_ts}.mp4")
                            try:
                                os.replace(self.rec_worker.out_path, new_path)
                                self.rec_worker.out_path = new_path
                            except Exception:
                                pass
                        msg = f"Recording saved: {self.rec_worker.out_path}" if self.rec_worker.out_path else "Recording stopped"
                        self.rec_worker = None
                        self._transition(self.state, msg)
                    else:
                        ensure_output_dir(self.config.video_output_dir)
                        self.rec_worker = ScreenRecorderWorker(bbox=None, output_dir=self.config.video_output_dir)
                        self.rec_worker.start()
                        self._transition(self.state, "Recording started")
                
                self._circle_prev = bool(circle)
                if self.freeze_window_open and self.frozen_bgr is not None:
                    freeze_overlay = self.frozen_bgr.copy()
                    if self.smoothed_index_xy:
                        cv2.circle(freeze_overlay, (int(self.smoothed_index_xy[0]), int(self.smoothed_index_xy[1])), 6, (0, 180, 255), -1)
                    if self.anchor_xy and self.current_xy:
                        x1, y1 = self.anchor_xy
                        x2, y2 = self.current_xy
                        x1, x2 = int(min(x1, x2)), int(max(x1, x2))
                        y1, y2 = int(min(y1, y2)), int(max(y1, y2))
                        if (x2 - x1) > 6 and (y2 - y1) > 6:
                            overlay_img = freeze_overlay.copy()
                            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=-1)
                            alpha = 0.25
                            cv2.addWeighted(overlay_img, alpha, freeze_overlay, 1 - alpha, 0, freeze_overlay)
                            cv2.rectangle(freeze_overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    hud_msg = "Frozen: pinch-drag, thumbs-up/c confirm, thumbs-down/fist/x cancel"
                    dbg = f"{hud_msg} | open:{int(open_palm)} pinch:{int(pinch)} fist:{int(fist)} up:{int(thumbs_up)} down:{int(thumbs_down)} circle:{int(circle)}"
                    cv2.putText(freeze_overlay, dbg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
                    cv2.imshow(self.freeze_window_name, freeze_overlay)

                if time.time() - self.last_action_ts > idle_timeout and self.state not in ("IDLE", "SELECTING", "LONG_CAPTURING"):
                    self.anchor_xy, self.current_xy = None, None
                    self._transition("IDLE", "Timeout -> IDLE")
                    self._exit_freeze()

                if self.config.preview:
                    display = self._draw_overlay(frame) if self.config.show_overlay else frame
                    cv2.imshow(window_name, display)
                    if time.time() % 2 < 0.1:
                        try:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                        except Exception:
                            pass
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord('a'):
                    if self.state == "IDLE":
                        self._enter_freeze()
                elif key == ord('f'):
                    if self.state == "IDLE":
                        self._enter_freeze()
                elif key == ord('c'):
                    if self.state in ("SELECTING", "ARMED"):
                        self._confirm_capture()
                elif key == ord('x'):
                    if self.state in ("ARMED", "SELECTING"):
                        self.anchor_xy, self.current_xy = None, None
                        self._transition("IDLE", "Cancelled")
                        self._exit_freeze()
                elif key == ord('l'):
                    if self.state != "LONG_CAPTURING":
                        bbox = self._selected_bbox()
                        if bbox is None:
                            self.status_text = "Select a region first (pinch) for long screenshot"
                        else:
                            self.long_worker = LongScreenshotWorker(bbox, self.config.output_dir)
                            self.long_worker.start()
                            self._transition("LONG_CAPTURING", "Long screenshot capturing... press 'l' to stop")
                    else:
                        if self.long_worker:
                            self.long_worker.stop()
                            self.long_worker.join(timeout=5.0)
                            out_path = self.long_worker.out_path
                            self.long_worker = None
                            self.anchor_xy, self.current_xy = None, None
                            if out_path:
                                self.last_snip_path = out_path
                                self._transition("IDLE", f"Long snip saved: {out_path}")
                            else:
                                self._transition("IDLE", "Long snip cancelled or failed")
                elif key == ord('r'):
                    if not self.rec_worker:
                        bbox = self._selected_bbox()
                        self.rec_worker = ScreenRecorderWorker(bbox=bbox, output_dir=self.config.output_dir)
                        self.rec_worker.start()
                        self._transition(self.state, "Recording started (press 'r' to stop)")
                    else:
                        self.rec_worker.stop()
                        self.rec_worker.join(timeout=5.0)
                        out_mp4 = self.rec_worker.out_path
                        self.rec_worker = None
                        if out_mp4:
                            self._transition(self.state, f"Recording saved: {out_mp4}")
                        else:
                            self._transition(self.state, "Recording stopped")
                elif key == ord('o'):
                    if pytesseract is None:
                        self.status_text = "pytesseract not installed"
                    elif not self.last_snip_path or not os.path.exists(self.last_snip_path):
                        self.status_text = "No snip available for OCR"
                    else:
                        try:
                            img = Image.open(self.last_snip_path)
                            text = pytesseract.image_to_string(img)
                            txt_path = os.path.splitext(self.last_snip_path)[0] + ".txt"
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(text)
                            self.status_text = f"OCR saved: {txt_path}"
                        except Exception as exc:
                            self.status_text = f"OCR failed: {exc}"
        finally:
            self.close()


class ScreenRecorderWorker(threading.Thread):
    def __init__(self, bbox: Optional[Tuple[int, int, int, int]], output_dir: str, fps: int = 20):
        super(ScreenRecorderWorker, self).__init__(daemon=True)
        self.bbox = bbox
        self.output_dir = output_dir
        self.fps = fps
        self.stop_event = threading.Event()
        self.out_path: Optional[str] = None
        self.start_ts: Optional[str] = None
        self.start_epoch: Optional[float] = None

    def run(self):
        if mss is None:
            return
        ensure_output_dir(self.output_dir)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        with mss.mss() as sct:
            if self.bbox:
                x1, y1, x2, y2 = self.bbox
                left = min(x1, x2)
                top = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                monitor = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
            else:
                mons = getattr(sct, "monitors", [])
                if isinstance(mons, list) and len(mons) >= 2:
                    mon = mons[1]
                elif isinstance(mons, list) and len(mons) == 1:
                    mon = mons[0]
                else:
                    mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
                monitor = {"left": mon.get("left", 0), "top": mon.get("top", 0),
                           "width": mon.get("width", 1920), "height": mon.get("height", 1080)}
                width, height = monitor["width"], monitor["height"]
            width = monitor["width"]
            height = monitor["height"]
            self.start_ts = now_timestamp()
            self.start_epoch = time.time()
            out_file = os.path.join(self.output_dir, f"{self.start_ts}.mp4")
            writer = cv2.VideoWriter(out_file, fourcc, self.fps, (width, height))
            try:
                last_time = time.time()
                frame_interval = 1.0 / float(self.fps)
                while not self.stop_event.is_set():
                    im = sct.grab(monitor)
                    frame = Image.frombytes("RGB", im.size, im.rgb)
                    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    try:
                        cv2.rectangle(frame, (4, 4), (width - 5, height - 5), (0, 255, 0), 6)
                        if self.start_epoch is not None:
                            elapsed = int(time.time() - self.start_epoch)
                            mm = elapsed // 60
                            ss = elapsed % 60
                            label = f"REC {mm:02d}:{ss:02d}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                            cv2.putText(frame, label, (max(10, width - tw - 20), max(th + 10, height - 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    except Exception:
                        pass
                    writer.write(frame)
                    dt = time.time() - last_time
                    if dt < frame_interval:
                        time.sleep(frame_interval - dt)
                    last_time = time.time()
                self.out_path = out_file
            finally:
                writer.release()

    def stop(self):
        self.stop_event.set()


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="HandSnip - Gesture-Driven Screen Snipping")
    parser.add_argument("--model", type=str, default=None, help="Path to Keras .h5 model or weights file")
    parser.add_argument("--frame_model", type=str, default=None, help="Path to 2D frame-based Keras .h5 model (e.g., MobileNetV2)")
    parser.add_argument("--frame_labels_json", type=str, default=None, help="Path to labels json produced during frame model training")
    parser.add_argument("--frame_input", type=int, default=128, help="Input size for frame model")
    parser.add_argument("--palm_frames", type=int, default=2, help="Consecutive open-palm frames required to freeze")
    parser.add_argument("--palm_spread", type=float, default=0.22, help="Finger spread threshold (0..1) for open-palm detection")
    parser.add_argument("--pinch_thresh", type=float, default=0.08, help="Thumb-index distance threshold (0..1) for pinch")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames per prediction window")
    parser.add_argument("--height", type=int, default=256, help="Frame height for model input")
    parser.add_argument("--width", type=int, default=256, help="Frame width for model input")
    parser.add_argument("--labels", type=str, default="dataset/labels_extracted.csv", help="Path to labels file")
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold for gesture acceptance")
    parser.add_argument("--smooth", type=int, default=5, help="Prediction smoothing window (votes)")
    parser.add_argument("--out", type=str, default="screenshots", help="Directory to save snips")
    parser.add_argument("--drag_gain", type=float, default=3.5, help="Scale factor for pinch-drag growth (>=1)")
    parser.add_argument("--edge_extrap_thresh", type=int, default=80, help="Px distance to edge where extrapolation engages")
    parser.add_argument("--edge_extrap_step", type=int, default=50, help="Px step per frame when extrapolating")
    parser.add_argument("--cam_norm_left", type=float, default=0.15, help="Camera active left bound (0..1)")
    parser.add_argument("--cam_norm_right", type=float, default=0.85, help="Camera active right bound (0..1)")
    parser.add_argument("--cam_norm_top", type=float, default=0.15, help="Camera active top bound (0..1)")
    parser.add_argument("--cam_norm_bottom", type=float, default=0.85, help="Camera active bottom bound (0..1)")
    parser.add_argument("--no-overlay", action="store_true", help="Disable drawing overlay on webcam preview")
    parser.add_argument("--preview", action="store_true", help="Show webcam preview window (otherwise runs background)")
    parser.add_argument("--video_out", type=str, default="video_recordings", help="Directory to save recordings")
    args = parser.parse_args()
    return RuntimeConfig(
        model_path=args.model,
        frame_model=args.frame_model,
        frame_labels_json=args.frame_labels_json,
        frame_input=args.frame_input,
        palm_frames=max(1, int(args.palm_frames)),
        palm_spread=float(args.palm_spread),
        pinch_dist_thresh=float(args.pinch_thresh),
        drag_gain=float(args.drag_gain),
        edge_extrapolate_thresh_px=int(args.edge_extrap_thresh),
        edge_extrapolate_step_px=int(args.edge_extrap_step),
        cam_norm_left=float(args.cam_norm_left),
        cam_norm_right=float(args.cam_norm_right),
        cam_norm_top=float(args.cam_norm_top),
        cam_norm_bottom=float(args.cam_norm_bottom),
        video_output_dir=str(args.video_out),
        double_palm_window_s=1.0,
        frames=args.frames,
        height=args.height,
        width=args.width,
        labels_path=args.labels,
        confidence_threshold=args.conf,
        smooth_window=args.smooth,
        output_dir=args.out,
        show_overlay=not args.no_overlay,
        preview=bool(args.preview)
    )


def load_labels(path: str) -> List[str]:
    if not os.path.exists(path):
        return [
            "Swiping Left",
            "Swiping Right",
            "Swiping Down",
            "Swiping Up",
            "Sliding Two Fingers Down",
            "Sliding Two Fingers Up",
            "Thumb Down",
            "Thumb Up",
            "Stop Sign",
            "No gesture",
        ]
    if not DATALOADER_AVAILABLE or DataLoader is None:
        print(f"WARNING: DataLoader not available. Cannot load labels from {path}. Using defaults.", file=sys.stderr)
        return [
            "Swiping Left",
            "Swiping Right",
            "Swiping Down",
            "Swiping Up",
            "Sliding Two Fingers Down",
            "Sliding Two Fingers Up",
            "Thumb Down",
            "Thumb Up",
            "Stop Sign",
            "No gesture",
        ]
    dl = DataLoader(path_labels=path)
    return dl.labels


def main():
    cfg = parse_args()
    labels = load_labels(cfg.labels_path)
    print(f"Loaded {len(labels)} labels (sequence model).")
    print("Key gestures (sequence/MediaPipe mode):")
    print(" - Stop Sign or Open Palm -> Arm | Keyboard: 'a'")
    print(" - Pinch (thumb-index) -> Draw/drag selection (MediaPipe)")
    print(" - Thumb Up or Circle -> Confirm | Keyboard: 'c'")
    print(" - Thumb Down -> Cancel | Keyboard: 'x'")
    print(" - Sliding Two Fingers Down/Up or pinch_drag -> Long screenshot start/stop | Keyboard: 'l' to toggle")
    print(" - Swiping Up/Down -> Recording start/stop | Keyboard: 'r' to toggle")
    print(" - Swiping Right -> OCR last snip | Keyboard: 'o'")
    app = HandSnipApp(cfg, labels)
    if cfg.frame_model and KERAS_AVAILABLE:
        frame_labels: List[str]
        if cfg.frame_labels_json and os.path.exists(cfg.frame_labels_json):
            import json
            with open(cfg.frame_labels_json, "r") as f:
                idx_map = json.load(f)
            frame_labels = [None] * len(idx_map)
            for name, idx in idx_map.items():
                frame_labels[idx] = name
            frame_labels = [str(x) for x in frame_labels]
        else:
            frame_labels = ["open_palm", "pinch_drag", "circle", "no_gesture"]
        try:
            app.frame_classifier = FrameGestureClassifier(cfg.frame_model, frame_labels, input_size=cfg.frame_input)
            print(f"Frame model loaded with labels: {frame_labels}")
        except Exception as exc:
            print(f"WARNING: Failed to load frame model: {exc}", file=sys.stderr)
    app.run()


if __name__ == "__main__":
    main()



