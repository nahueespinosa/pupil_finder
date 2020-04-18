import argparse
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test CNN to predict pupil position using a camera.')
    parser.add_argument('filename', type=str, help='name of the input model file')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.filename)

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eyes = cv2.CascadeClassifier('cascades/haarcascade_eye.xml').detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5
        )

        pupil = (0, 0)

        for (x, y, h, w) in eyes:
            roi = gray_frame[y: y + h, x: x + w]
            roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
            roi = np.reshape(roi, (128, 128, 1))
            result = model.predict(np.array([roi]))
            pupil = (int(result[0][0])+x, int(result[0][1])+y)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            break

        cv2.circle(frame, pupil, 2, (255, 0, 0), 2)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
