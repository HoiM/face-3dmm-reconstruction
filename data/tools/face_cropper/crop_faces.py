import os
import cv2
import dlib
import numpy as np

from .align_trans import warp_and_crop_face


class FaceCropper:
    def __init__(self):
        curr_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.detector_path = os.path.join(curr_dir_path, "shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.detector_path)
        self.reference_points = np.asarray([
            [38.29459953, 51.69630051],
            [73.53179932, 51.50139999],
            [56.02519989, 71.73660278],
            [41.54930115, 92.36550140],
            [70.72990036, 92.20410156]
        ])

    def crop(self, image_path, save_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        face = faces[0]
        landmarks = self.predictor(gray, face)
        landmarks = [[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)]
        landmarks = np.asarray(landmarks)
        landmarks_5 = self.convert_68_to_5(landmarks)
        warped_face, _ = warp_and_crop_face(
            src_img=img,
            facial_pts=landmarks_5,
            reference_pts=self.reference_points,
            crop_size=(256, 256),
            return_trans_inv=True
        )
        cv2.imwrite(save_path, warped_face)

    def cal_68_landmarks(self, image_path, save_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        face = faces[0]
        landmarks = self.predictor(gray, face)
        landmarks = [[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)]
        landmarks = np.asarray(landmarks)
        np.save(save_path, landmarks)


    @staticmethod
    def convert_68_to_5(landmarks68):
        res = np.zeros([5, 2])
        res[0, :] = (landmarks68[36, :] + landmarks68[39, :]) / 2
        res[1, :] = (landmarks68[42, :] + landmarks68[45, :]) / 2
        res[2, :] = landmarks68[33, :]
        res[3, :] = landmarks68[48, :]
        res[4, :] = landmarks68[54, :]
        return res


if __name__ == "__main__":
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    example_image_path = os.path.join(curr_dir_path, "example.jpg")
    save_path = os.path.join(curr_dir_path, "cropped.jpg")

    cropper = FaceCropper()
    cropper(example_image_path, save_path)
