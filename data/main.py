import os
from tools import FaceCropper, Segmentor


def main():
    images_dir = "images"
    data_dir = "data"
    device = "cuda:0"
    cropper = FaceCropper()
    segmentor = Segmentor(device)
    for fn in os.listdir(images_dir):
        name = fn.split(".")[0]
        fp = os.path.join(images_dir, fn)
        cropped_face_save_path = os.path.join(data_dir, name + "_cropped.jpg")
        landmark_save_path = os.path.join(data_dir, name + "_landmarks.npy")
        mask_save_path = os.path.join(data_dir, name + "_mask.npy")
        cropper.crop(fp, cropped_face_save_path)
        cropper.cal_68_landmarks(cropped_face_save_path, landmark_save_path)
        segmentor.segment(cropped_face_save_path, mask_save_path)

if __name__ == '__main__':
    main()
