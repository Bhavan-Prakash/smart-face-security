import cv2
import numpy as np


cap = cv2.VideoCapture("/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/faces_testing.mp4")

detector = cv2.FaceDetectorYN.create("/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/models/face_detection_yunet_2026may.onnx",
                                      "",
                               (320,320))




def detection(frame):
    h, w = frame.shape[:2]

    detector.setInputSize((w,h))

    _, face = detector.detect(frame)
    return face

def landmaek_generation()


while(True):
    ret, frame = cap.read()

    if not ret:
        break

    detect = detection(frame)
    print(detect)

    cv2.imshow("tracker", frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()