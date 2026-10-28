# import cv2
# import time
# import numpy as np
# from pathlib import Path
# from ultralytics.trackers.byte_tracker import BYTETracker
# from types import SimpleNamespace
# import time
# import threading
# import queue
# import requests

# class LatestFrameReader:
#     def __init__(self, src, retries=5, delay=2):
#         self.src = src
#         self.retries = retries
#         self.delay = delay
#         self.cap = self._open()
#         self.q = queue.Queue(maxsize=1)
#         self.stopped = False
#         self.thread = threading.Thread(target=self._reader, daemon=True)
#         self.thread.start()

#     def _open(self):
#         for _ in range(self.retries):
#             cap = cv2.VideoCapture(self.src)
#             if cap.isOpened():
#                 return cap
#             time.sleep(self.delay)
#         return None

#     def _reader(self):
#         while not self.stopped:
#             if self.cap is None or not self.cap.isOpened():
#                 print("Stream lost, reconnecting...")
#                 if self.cap is not None:
#                     self.cap.release()
#                 self.cap = self._open()
#                 if self.cap is None:
#                     time.sleep(5)
#                 continue

#             ret, frame = self.cap.read()

#             if not ret:
#                 self.cap.release()
#                 self.cap = None
#                 continue

#             if not self.q.empty():
#                 try:
#                     self.q.get_nowait()
#                 except queue.Empty:
#                     pass
#             self.q.put(frame)

#     def read(self, timeout=5):
#         try:
#             return True, self.q.get(timeout=timeout)
#         except queue.Empty:
#             return False, None

#     def release(self):
#         self.stopped = True
#         self.thread.join(timeout=2)
#         if self.cap is not None:
#             self.cap.release()

# # cap = cv2.VideoCapture("")

# # cap = cv2.VideoCapture("/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/faces_testing.mp4")

# detector = cv2.FaceDetectorYN.create(
#     "/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/models/face_detection_yunet_2026may.onnx",
#     "",
#     (320, 320)
# )

# RECOGNITION_THRESHOLD = 0.45

# recognizer = cv2.dnn.readNetFromONNX("/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/models/w600k_mbf.onnx")

# EMBEDDINGS_DIR = Path(
#     "/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/embeddings_for_known_faces"
# )

# def display_info(frame, fps):
#     height, width = frame.shape[:2]

#     # Panel dimensions
#     panel_width = 300
#     panel_height = 100

#     # Position: middle of the left side
#     x1 = 10
#     y1 = (height - panel_height) // 2

#     x2 = x1 + panel_width
#     y2 = y1 + panel_height

#     # Background panel
#     cv2.rectangle(
#         frame,
#         (x1, y1),
#         (x2, y2),
#         (0, 0, 0),
#         -1
#     )

#     # FPS
#     cv2.putText(
#         frame,
#         f"FPS: {fps:.1f}",
#         (x1 + 10, y1 + 35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (0, 255, 0),
#         2
#     )

#     # Resolution
#     cv2.putText(
#         frame,
#         f"Resolution: {width}x{height}",
#         (x1 + 10, y1 + 75),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (255, 255, 255),
#         2
#     )

#     return frame

# previous_time = time.time() # part of the fps we are defining on frame


# ARC_FACE_TEMPLATE = np.array([
#     [38.2946, 51.6963],
#     [73.5318, 51.5014],
#     [56.0252, 71.7366],
#     [41.5493, 92.3655],
#     [70.7299, 92.2041]
# ], dtype=np.float32)


# def preprocess_face(frame, landmarks):
#     # 1. Align face

#     transformation_matrix, _ = cv2.estimateAffinePartial2D(
#         landmarks,
#         ARC_FACE_TEMPLATE
#     )

#     face = cv2.warpAffine(
#         frame,
#         transformation_matrix,
#         (112, 112)
#     )

#     # 2. BGR -> RGB


#     face = cv2.cvtColor(
#         face,
#         cv2.COLOR_BGR2RGB
#     )

#     # 3. Convert uint8 -> float32


#     face = face.astype(np.float32)

#     # 4. Normalize to approximately [-1, 1]

#     face = (face - 127.5)/127.5

#     # 5. HWC -> CHW

#     face = face.transpose(2,0,1)

#     # 6. Add batch dimension

#     face = np.expand_dims(face, axis=0)

#     return face

# def detection(frame):
#     height, width = frame.shape[:2]

#     detector.setInputSize((width, height))

#     _, faces = detector.detect(frame)

#     return faces

# def detection_box_to_xyxy(face):
#     x, y, w, h = face[:4]

#     return [
#         x,
#         y,
#         x + w,
#         y + h
#     ]

# def prepare_tracker_results(faces):
#     if faces is None or len(faces) == 0:
#         return TrackerResults(
#             np.empty((0,4), dtype= np.float32),
#             np.empty((0,), dtype = np.float32),
#             np.empty((0,), dtype = np.float32)
#         )

#     xywh = []
#     conf = []

#     for face in faces:
#         x,y,w,h = face[:4]

#         cx = x + w / 2
#         cy = y + h / 2

#         xywh.append([cx,cy,w,h])
#         conf.append(float(face[-1]))

#     xywh = np.array(xywh, dtype=np.float32)
#     conf = np.array(conf, dtype=np.float32)
#     cls = np.zeros(len(conf), dtype=np.float32)

#     return TrackerResults(xywh, conf, cls)

# class TrackerResults:
#     def __init__(self, xywh, conf, cls):
#         self.xywh = xywh
#         self.conf = conf
#         self.cls = cls

#     def __len__(self):
#         return len(self.conf)

#     def __getitem__(self, mask):
#         return TrackerResults(
#             self.xywh[mask],
#             self.conf[mask],
#             self.cls[mask]
#         )

# def bounding_box(img, faces):
#     if faces is not None:
#         for face in faces:
#             x, y, w, h = face[:4]

#             img = cv2.rectangle(
#                 img,
#                 (int(x), int(y)),
#                 (int(x+w), int(y+h)),
#                 (0,255,0),
#                 4
#             )

#     return img

# def landmark_extraction(faces):
#     landmark_list = []

#     if faces is not None:
#         for face in faces:
#             landmarks = np.array([
#                 [face[4], face[5]],    # right eye
#                 [face[6], face[7]],    # left eye
#                 [face[8], face[9]],    # nose
#                 [face[10], face[11]],  # right mouth
#                 [face[12], face[13]],  # left mouth
#             ],dtype = np.float32)

#             landmark_list.append(landmarks)
#     return landmark_list

# def query_embeddings(processed_frame):
#     recognizer.setInput(processed_frame)
#     embedding = recognizer.forward()
#     # L2 normalization
#     embedding = embedding / np.linalg.norm(embedding)
#     return embedding

# def recognize(query_embedding, known_embeddings):

#     best_person = None
#     best_similarity = -1

#     # Remove batch dimension
#     query_embedding = query_embedding.flatten()

#     for person_name, person_embeddings in known_embeddings.items():

#         similarities = np.dot(
#             person_embeddings,
#             query_embedding
#         )

#         person_similarity = np.max(
#             similarities
#         )

#         if person_similarity > best_similarity:

#             best_person = person_name
#             best_similarity = person_similarity

#     return best_person, best_similarity

# def calculate_iou(box_a, box_b):
#     # box format:
#     # [x1, y1, x2, y2]

#     # Find the coordinates of the intersection rectangle
#     x1 = max(box_a[0], box_b[0])
#     y1 = max(box_a[1], box_b[1])

#     x2 = min(box_a[2], box_b[2])
#     y2 = min(box_a[3], box_b[3])

#     # Width and height of intersection
#     intersection_width = max(0, x2 - x1)
#     intersection_height = max(0, y2 - y1)

#     intersection_area = (
#         intersection_width * intersection_height
#     )

#     # Area of both boxes
#     box_a_width = box_a[2] - box_a[0]
#     box_a_height = box_a[3] - box_a[1]

#     box_b_width = box_b[2] - box_b[0]
#     box_b_height = box_b[3] - box_b[1]

#     area_a = box_a_width * box_a_height
#     area_b = box_b_width * box_b_height

#     # Union
#     union_area = area_a + area_b - intersection_area

#     if union_area == 0:
#         return 0

#     iou = intersection_area / union_area

#     return iou

# def match_track_to_detection(track_box, detections):
    
#     best_detection_index = None
#     best_iou = 0.3

#     for i, face in enumerate(detections):

#         detection_box = detection_box_to_xyxy(face)

#         iou = calculate_iou(
#             track_box,
#             detection_box
#         )

#         if iou > best_iou:
#             best_iou = iou
#             best_detection_index = i

#     return best_detection_index, best_iou

# #below is the embedding comparision code
# known_embeddings = {}

# for embedding_file in EMBEDDINGS_DIR.glob("*.npy"):

#     person_name = embedding_file.stem

#     embeddings = np.load(embedding_file)

#     known_embeddings[person_name] = embeddings

#     # print(
#     #     person_name,
#     #     embeddings.shape
#     # )



# tracker_args = SimpleNamespace(
#     track_high_thresh=0.5,
#     track_low_thresh=0.1,
#     new_track_thresh=0.6,
#     track_buffer=30,
#     match_thresh=0.8,
#     fuse_score=True
# )

# tracker = BYTETracker(tracker_args)
# # TRACK_MEMORY_TIMEOUT = 1 #1 is not recommended just for testing
# TRACK_MEMORY_TIMEOUT_SECONDS = 1.0

# RECOGNITION_INTERVAL = 2.0 # Re-recognize an active track every 2 seconds.

# frame_number = 0

# track_memory = {}
# recognition_count = 0

# cap = LatestFrameReader("rtsp://")



# # ============================================================
# # BACKEND API
# # ============================================================

# # ============================================================
# # BACKEND API
# # ============================================================

# BACKEND_URL = "http://127.0.0.1:8000"
# CAMERA_ID = "entrance"

# def _send_recognition_event(
#     person_name,
#     similarity,
#     track_id
# ):

#     payload = {
#         "camera_id": CAMERA_ID,
#         "person_name": person_name,
#         "similarity": float(similarity),
#         "track_id": int(track_id)
#     }

#     try:

#         response = requests.post(
#             f"{BACKEND_URL}/api/recognitions",
#             json=payload,
#             timeout=1
#         )

#         if response.status_code == 200:

#             print(
#                 f"[API] Recognition sent: "
#                 f"{person_name} "
#                 f"(track={track_id}, "
#                 f"similarity={similarity:.3f})"
#             )

#         else:

#             print(
#                 f"[API] Failed: "
#                 f"{response.status_code}"
#             )

#     except requests.RequestException as e:

#         print(
#             f"[API] Backend unavailable: {e}"
#         )

# def send_recognition_event(
#     person_name,
#     similarity,
#     track_id
# ):

#     thread = threading.Thread(
#         target=_send_recognition_event,
#         args=(
#             person_name,
#             similarity,
#             track_id
#         ),
#         daemon=True
#     )

#     thread.start()


# while True:
#     ret, frame = cap.read()

#     if not ret:
#         # queue was empty for 5s straight — reader thread is
#         # still trying to reconnect in the background
#         continue

#     frame_number += 1


#     #pipeline
#     detect = detection(frame)
#     landmark_list = landmark_extraction(detect)
#     tracker_result = prepare_tracker_results(detect)
    
#     tracks = tracker.update(
#         tracker_result,
#         frame
#     )

#     for track in tracks:

#         x1, y1, x2, y2 = track[:4]

#         track_id = int(track[4])

#         track_box = [
#             x1,
#             y1,
#             x2,
#             y2
#         ]

#         detection_index, iou = match_track_to_detection(
#             track_box,
#             detect
#         )

#         if detection_index is None:
#             continue

#         landmarks = landmark_list[detection_index]

#         # print(
#         #     f"Track {track_id} → "
#         #     f"Detection {detection_index} → "
#         #     f"Landmarks:\n{landmarks}"
#         # )

#         if track_id not in track_memory:

#             # print(
#             #     f"\nRecognizing Track ID: {track_id}"
#             # )

#             prep_face = preprocess_face(
#                 frame,
#                 landmarks
#             )

#             recognition_count += 1

#             query_embed = query_embeddings(
#                 prep_face
#             )

#             person, similarity = recognize(
#                 query_embed,
#                 known_embeddings
#             )

#             if similarity >= RECOGNITION_THRESHOLD:
#                 name = person
#             else:
#                 name = "Unknown"

#             track_memory[track_id] = {
#                 "name": name,
#                 "similarity": similarity,
#                 # "last_seen": frame_number,
#                 "last_recognition_time": time.time(),
#                 "last_seen_time": time.time()
#             }

#             # ========================================================
#             # SEND NEW RECOGNITION TO BACKEND
#             # ========================================================

#             send_recognition_event(
#                 person_name=name,
#                 similarity=similarity,
#                 track_id=track_id
#             )

#         else:

#             name = track_memory[track_id]["name"]
#             similarity = track_memory[track_id]["similarity"]

#             # Track is still alive
#             # track_memory[track_id]["last_seen"] = frame_number
#             track_memory[track_id]["last_seen_time"] = time.time()
#             current_time = time.time()
#             if current_time - track_memory[track_id]["last_recognition_time"] >= RECOGNITION_INTERVAL:
#                 prep_face = preprocess_face(frame, landmarks)
#                 recognition_count += 1
#                 query_embed = query_embeddings(prep_face)
#                 person, similarity = recognize(query_embed, known_embeddings)

#                 if similarity >= RECOGNITION_THRESHOLD:
#                     name = person
#                 else:
#                     name = "Unknown"

#                 track_memory[track_id]["name"] = name
#                 track_memory[track_id]["similarity"] = similarity
#                 track_memory[track_id]["last_recognition_time"] = current_time

#         # ====================================================
#         # DRAW FACE BOUNDING BOX
#         # ====================================================

#         cv2.rectangle(

#             frame,

#             (
#                 int(x1),
#                 int(y1)
#             ),

#             (
#                 int(x2),
#                 int(y2)
#             ),

#             (0, 255, 0),

#             3
#         )

#         # ====================================================
#         # DRAW TRACK / RECOGNITION INFORMATION
#         # ====================================================

#         label_y = max(
#             int(y1) - 60,
#             25
#         )

#         cv2.putText(

#             frame,

#             f"{name}",

#             (
#                 int(x1),
#                 label_y
#             ),

#             cv2.FONT_HERSHEY_SIMPLEX,

#             0.7,

#             (0, 255, 0),

#             2
#         )

#         cv2.putText(

#             frame,

#             f"Similarity: {similarity:.2f}",

#             (
#                 int(x1),
#                 label_y + 22
#             ),

#             cv2.FONT_HERSHEY_SIMPLEX,

#             0.55,

#             (255, 255, 255),

#             2
#         )

#         cv2.putText(

#             frame,

#             f"Track ID: {track_id}",

#             (
#                 int(x1),
#                 label_y + 44
#             ),

#             cv2.FONT_HERSHEY_SIMPLEX,

#             0.55,

#             (255, 255, 255),

#             2
#         )

#     stale_tracks = []

#     for track_id, data in track_memory.items():

#         # frames_missing = frame_number - data["last_seen"]

#         # if frames_missing > TRACK_MEMORY_TIMEOUT:
#         #     stale_tracks.append(track_id)

#         if time.time() - data["last_seen_time"] > TRACK_MEMORY_TIMEOUT_SECONDS:
#             stale_tracks.append(track_id)


#     for track_id in stale_tracks:

#         print(
#             f"Removing expired Track ID: {track_id}"
#         )

#         del track_memory[track_id]

        
#     # ========================================================
#     # FPS
#     # ========================================================

#     current_time = time.time()

#     fps = 1 / (
#         current_time -
#         previous_time
#     )

#     previous_time = current_time

#     # ========================================================
#     # GLOBAL DEBUG INFORMATION
#     # ========================================================

#     cv2.putText(

#         frame,

#         f"FPS: {fps:.1f}",

#         (20, 35),

#         cv2.FONT_HERSHEY_SIMPLEX,

#         0.7,

#         (0, 255, 0),

#         2
#     )

#     cv2.putText(

#         frame,

#         f"Recognition Frames: {recognition_count}",

#         (20, 65),

#         cv2.FONT_HERSHEY_SIMPLEX,

#         0.7,

#         (0, 255, 255),

#         2
#     )

#     cv2.putText(

#         frame,

#         f"Frame: {frame_number}",

#         (20, 95),

#         cv2.FONT_HERSHEY_SIMPLEX,

#         0.7,

#         (255, 255, 255),

#         2
#     )

       
#     cv2.imshow("Camera feed", frame)

    

#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()




"""
main.py - program entry point.

Usage:
    python main.py                     run every enabled camera in cameras.yaml
    python main.py --camera entrance   run only the camera with this id
    python main.py --headless          no preview windows (for servers)

How it works: the main process is a "supervisor". It starts one separate
process per camera (so cameras run in parallel on different CPU cores) and
restarts any camera process that crashes.
"""
import argparse
import logging
import multiprocessing as mp
import sys
import time

import cv2

from config import settings
from camera_source import load_cameras, open_source
from face_engine import FaceEngine
from pipeline import CameraPipeline

LOG_FORMAT = "%(asctime)s [%(processName)s] %(levelname)s %(name)s: %(message)s"


def draw_overlay(frame, cam, fps, recognitions):
    """Draw the small info panel (camera name, FPS, ...) in the top-left corner."""
    h, w = frame.shape[:2]
    lines = [
        (cam.name, (255, 255, 255)),
        (f"FPS: {fps:.1f}", (0, 255, 0)),
        (f"Recognitions: {recognitions}", (0, 255, 255)),
        (f"{w}x{h}", (255, 255, 255)),
    ]
    cv2.rectangle(frame, (10, 10), (260, 20 + 28 * len(lines)), (0, 0, 0), -1)  # black panel
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, 38 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def handle_event(event):
    """
    Called whenever a person is recognised / an unknown face is confirmed.

    For now it only writes a log line. LATER: send `event` to your backend
    here (HTTP POST, database insert, push notification, ...).
    """
    logging.getLogger("event").info(
        "%s | camera=%s | track=%s | similarity=%.2f",
        event["person_name"], event["camera_id"], event["track_id"], event["similarity"],
    )


def run_camera(cam, show):
    """Runs inside its own process: one camera, from connection to shutdown."""
    # Child processes start fresh, so logging must be configured again here.
    logging.basicConfig(level=settings.log_level, format=LOG_FORMAT)
    log = logging.getLogger("camera")

    # Each process loads its own models (they can't be shared between processes).
    engine = FaceEngine(settings)
    pipeline = CameraPipeline(cam, settings, engine, on_event=handle_event)
    reader = open_source(cam)

    fps, prev = 0.0, time.time()
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                if reader.ended:           # a non-looping video file finished
                    log.info("[%s] video ended", cam.id)
                    break
                continue                   # stream is down; reader is reconnecting

            pipeline.process(frame, draw=show)  # detect -> track -> recognise -> draw

            # Smoothed FPS (90% old value + 10% new) so the number doesn't jitter.
            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            prev = now

            if show:
                draw_overlay(frame, cam, fps, pipeline.recognition_count)
                cv2.imshow(cam.name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # press q in the window to quit
                    break
    except KeyboardInterrupt:  # Ctrl+C
        pass
    finally:
        reader.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face recognition security system")
    parser.add_argument("--cameras", default=str(settings.cameras_file), help="path to cameras.yaml")
    parser.add_argument("--camera", action="append", help="only run this camera id (can repeat)")
    parser.add_argument("--headless", action="store_true", help="do not open preview windows")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format=LOG_FORMAT)
    log = logging.getLogger("supervisor")

    # Keep only enabled cameras (and only the requested ones, if --camera was used).
    cameras = [c for c in load_cameras(args.cameras)
               if c.enabled and (not args.camera or c.id in args.camera)]
    if not cameras:
        sys.exit("No enabled cameras. Edit cameras.yaml and set enabled: true.")

    show = settings.show_window and not args.headless

    # "spawn" behaves the same on macOS, Windows and Linux.
    ctx = mp.get_context("spawn")
    procs = {}  # camera id -> (CameraConfig, Process)

    def start(cam):
        p = ctx.Process(target=run_camera, args=(cam, show), name=cam.id)
        p.start()
        procs[cam.id] = (cam, p)
        log.info("Started camera '%s' (%s)", cam.id, cam.safe_source)

    for cam in cameras:
        start(cam)

    # Supervisor loop: watch the camera processes.
    try:
        while procs:
            time.sleep(1)
            for cid, (cam, p) in list(procs.items()):
                if p.is_alive():
                    continue
                if p.exitcode == 0:  # clean exit (user pressed q / video ended) - don't restart
                    log.info("Camera '%s' finished", cid)
                    del procs[cid]
                else:                # crashed - wait a little, then restart it
                    log.error("Camera '%s' crashed (exit code %s) - restarting in 5s", cid, p.exitcode)
                    time.sleep(5)
                    start(cam)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        for _, p in procs.values():
            p.terminate()
        for _, p in procs.values():
            p.join(timeout=5)


# This guard is REQUIRED when using multiprocessing with "spawn".
if __name__ == "__main__":
    main()