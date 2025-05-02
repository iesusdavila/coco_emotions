import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

class FacialExpressionModel(object):

    mapper = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'happiness',
        4: 'sadness',
        5: 'surprise',
        6: 'neutral'
    }


    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy",
                    "Sad", "Surprise", "Neutral"]

    def __init__(self, model):
        self.model=model

    def predict_emotion(self, img):
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        self.preds = self.model.predict(img, verbose=0)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

class VideoCamera(object):
    def __init__(self,path):
        self.video = cv2.VideoCapture(path)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        _, fr = self.video.read()

        rgb_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = fr.shape
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]

                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                expand = 45
                x_min = max(0, x_min - expand)
                y_min = max(0, y_min - expand)
                x_max = min(w, x_max + expand)
                y_max = min(h, y_max + expand)

                fc = rgb_frame[y_min:y_max, x_min:x_max]
                roi = cv2.resize(fc, (48, 48))

                cv2.imshow('ROI', roi)

                if roi.size > 0:
                    pred = emotion_model.predict_emotion(roi)
                cv2.putText(fr, pred, (x_min, y_min), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x_min,y_min),(x_max,y_max),(255,0,0),2)
        return fr


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5)

model = tf.keras.models.load_model('model.h5')
emotion_model = FacialExpressionModel(model)
font = cv2.FONT_HERSHEY_SIMPLEX

def gen(camera):
    while True:
        frame = camera.get_frame()
        cv2.imshow('frame', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = VideoCamera(0)
    gen(camera)



