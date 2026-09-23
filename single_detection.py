import cv2

# ONNX = Open Neural Network Exchange.
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2026may.onnx", 
    "",
    (320,320)
)

img = cv2.imread("/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/single_face.jpeg")
print(img.shape) #(1280, 960, 3)

height, width = img.shape[0:2]
detector.setInputSize((width, height))

_, faces = detector.detect(img)

# print(faces)

if faces is not None:

    for face in faces:
        x, y, w, h = face[0:4]

        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,255), 3)

cv2.imshow("Face Detector", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
