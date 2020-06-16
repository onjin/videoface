#!/usr/bin/env python
import argparse

import numpy as np
import cv2


def main(params):
    print("Press 'q' to exit")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

    if params.smiles:
        smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

    if params.eyes:
        eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

        # rectangle face
        for face_nr, (x, y, w, h) in enumerate(faces, 1):
            cv2.rectangle(img, (x, y), (x + w, y + h), (190, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img[y : y + h, x : x + w]

            if params.eyes:
                eyes = eye_cascade.detectMultiScale(roi_color)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 190, 0), 2
                    )

            if params.smiles:
                smile = smile_cascade.detectMultiScale(roi_gray, 1.5, 15)
                for (sx, sy, sw, sh) in smile:
                    cv2.rectangle(
                        roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 190), 2
                    )

            if params.blur >= face_nr:
                face = img[y : y + h, x : x + w]
                face = cv2.blur(face, ((w // 5), (h // 5)))
                img[y : y + h, x : x + w] = face

        cv2.imshow("img", img)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="videofacce")
    parser.add_argument(
        "-b", "--blur", default=0, action="store", type=int, help="blur N faces"
    )
    parser.add_argument(
        "-e", "--eyes", default=False, action="store_true", help="detect eyes"
    )
    parser.add_argument(
        "-s", "--smiles", default=False, action="store_true", help="detect smiles"
    )
    params = parser.parse_args()

    main(params)
