import cv2
import numpy as np

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2026may.onnx",
    "",
    (320,320)
)

recognizer = cv2.dnn.readNetFromONNX(
    "w600k_mbf.onnx",
)

img1 = cv2.imread("/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/img1_nan.jpg")
img2 = cv2.imread("/Users/bhavan/Documents/codes/Home_Security_Face_Recognition/img2_nan.jpg")

ARC_FACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def detect(frame):
    height, width = frame.shape[:2]

    detector.setInputSize((width,height))

    _, faces = detector.detect(frame)

    return faces

def draw_boundingBox(frame, faces):
    if faces is not None:
            for face in faces:
                
                x, y, w, h = face[:4]
    
                cv2.rectangle(
                    frame,
                    (int(x), int(y)),
                    (int(x+w), int(y+h)),
                    (255, 0, 0),
                    4
                )
    
                return frame

def preprocess(frame, landmarks):
    transformation, _ = cv2.estimateAffinePartial2D(
        landmarks,
        ARC_FACE_TEMPLATE
    )

    face = cv2.warpAffine(
        frame,
        transformation,
        (112,112)
    )

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face = face.astype(np.float32)

    face = (face-127.5)/127.5

    face = face.transpose(2,0,1)

    face = np.expand_dims(face, axis=0)

    return face

def extract_landmarks(faces):
    if faces is not None:
         for face in faces:  
            landmarks = np.array([
                [face[4], face[5]],
                [face[6], face[7]],
                [face[8], face[9]],
                [face[10], face[11]],
                [face[12], face[13]]
            ], dtype=np.float32)

            return landmarks

def embedding(preprocessed_frame):
     recognizer.setInput(preprocessed_frame)

     embedding = recognizer.forward()
     return embedding
    
faces1 = detect(img1)

landmarks1 = extract_landmarks(faces1)

if landmarks1 is not None:
    processed_face1 = preprocess(img1, landmarks1)
    embedding1 = embedding(processed_face1)

    norm_embeddings1 = embedding1 / np.linalg.norm(embedding1)

    print("Embedding 1 shape:", norm_embeddings1.shape)
    print("Embedding 1 norm:", np.linalg.norm(norm_embeddings1))


faces2 = detect(img2)

landmarks2 = extract_landmarks(faces2)

if landmarks2 is not None:
    processed_face2 = preprocess(img2, landmarks2)

    embedding2 = embedding(processed_face2)

    norm_embeddings2 = embedding2 / np.linalg.norm(embedding2)

    print("Embedding 2 shape:", norm_embeddings2.shape)
    print("Embedding 2 norm:", np.linalg.norm(norm_embeddings2))


similarity = np.dot(
    norm_embeddings1.flatten(),
    norm_embeddings2.flatten()
)
print("Cosine similarity:", similarity)

cv2.waitKey(0)
cv2.destroyAllWindows


