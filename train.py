import os
import cv2
import torch
import torch.utils.data
import torch.utils.tensorboard
import PIL.Image
import numpy as np

from datasets import get_train_val_loaders
from deep_3d_face import (
    get_encoder,
    Decoder,
    Renderer,
    PerceptualLoss,
    LandmarkLoss,
    PhotometricLoss,
    CoefficientRegularizationLoss
)


# data
bfm_params_dir = "params/bfm"
data_dir = "data/data"
batch_size = 32
num_workers = 2
# optimizer
learning_rate = 1e-4
weight_decay = 0
# training & info
num_epochs = 500
print_iter = 20
val_iter = 2000
summary_dir = "ckpts/summary"
# resume training related
decoder_trainable = False
encoder_fixed = False
decoder_fixed = False
resume_encoder_params = ""
resume_decoder_params = ""
vis_dir = "ckpts/images"
save_encoder_path = "ckpts/encoder_%03d.pth"
save_decoder_path = "ckpts/decoder_%03d.pth"


def save_images(original, rendered, gt_landmarks, pred_landmarks, epoch, iteration, save_dir):
    """
    :param original: (bs, 3, h, w) (-1, 1) RGB GPU
    :param rendered: (bs, 3, h, w) (-1, 1) RGB GPU
    :param gt_landmarks: (bs, 68, 2) GPU
    :param pred_landmarks: (bs, 68, 2) GPU
    :param epoch:
    :param iteration:
    :param save_dir: directory to save images
    :return: image: (numpy.ndarray, numpy.uint8, 0-255) an integrated image
    """
    bs, c, h, w = original.shape
    ori = original * 127.5 + 127.5
    ren = rendered * 127.5 + 127.5
    ori = ori.cpu().detach().numpy().transpose(0, 2, 3, 1)
    ren = ren.cpu().detach().numpy().transpose(0, 2, 3, 1)
    gt_ldmk = gt_landmarks.cpu().detach().numpy().reshape((-1, 68, 2))
    pd_ldmk = pred_landmarks.cpu().detach().numpy().reshape((-1, 68, 2))
    length = bs if bs < 8 else 8
    image = np.zeros([h * 4, w * length, c], dtype=np.uint8)
    for i in range(length):
        ori_cp = np.ascontiguousarray(ori[i].copy(), dtype=np.uint8)
        ren_cp = np.ascontiguousarray(ren[i].copy(), dtype=np.uint8)
        image[0:h, i*w:(i+1)*w, :] = ori_cp
        image[h:2*h, i*w:(i+1)*w, :] = ren_cp
        for j in range(68):
            ori_cp = cv2.circle(ori_cp, (gt_ldmk[i][j][0], gt_ldmk[i][j][1]),
                                radius=1, color=(255, 255, 255), thickness=1)
            ren_cp = cv2.circle(ren_cp, (pd_ldmk[i][j][0], pd_ldmk[i][j][1]),
                                radius=1, color=(255, 255, 255), thickness=1)
        image[2 * h:3 * h, i * w:(i + 1) * w, :] = ori_cp
        image[3 * h:4 * h, i * w:(i + 1) * w, :] = ren_cp
    image_pil = PIL.Image.fromarray(image)
    if type(epoch) is int:
        image_pil.save(os.path.join(save_dir, "%d_%d.png" % (epoch, iteration)))
    else:
        image_pil.save(os.path.join(save_dir, "val_%d.png" % iteration))
    return image


def main():
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    # dataset
    train_loader, val_loader = get_train_val_loaders(data_dir, batch_size, num_workers)
    train_length = len(train_loader)
    global_val_iteration = 0
    # tensorboard
    writer = torch.utils.tensorboard.SummaryWriter(summary_dir, flush_secs=30)
    # networks
    encoder = get_encoder()
    decoder = Decoder(bfm_params_dir=bfm_params_dir, trainable=decoder_trainable)
    if resume_encoder_params:
        encoder.load_state_dict(torch.load(resume_encoder_params))
    if decoder_trainable and resume_decoder_params:
        decoder.load_state_dict(torch.load(resume_decoder_params))
    renderer = Renderer()
    # move to cuda
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    renderer = renderer.cuda()
    # losses
    photometric_loss = PhotometricLoss()
    landmark_loss = LandmarkLoss()
    perceptual_loss = PerceptualLoss()
    coefficient_regularization_loss = CoefficientRegularizationLoss()
    photometric_loss = photometric_loss.cuda()
    landmark_loss = landmark_loss.cuda()
    perceptual_loss = perceptual_loss.cuda()
    coefficient_regularization_loss = coefficient_regularization_loss.cuda()
    # optimizer
    parameters = list()
    if not encoder_fixed:
        parameters.extend(encoder.parameters())
    if decoder_trainable and (not decoder_fixed):
        parameters.extend(decoder.parameters())
    optimizer = torch.optim.Adam(params=parameters,
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    # train & val
    for epoch in range(num_epochs):
        for iteration, (images, landmarks, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            encoder.train()
            decoder.train()
            renderer.train()
            # data
            images = images.cuda()
            landmarks = landmarks.cuda()
            masks = masks.cuda()
            # forward
            coeff = encoder(images)
            face_projection, face_color, landmarks_3d, tri = decoder(coeff)
            rendered_images, _ = renderer(face_projection, face_color, tri)
            # loss
            loss_photometric = photometric_loss(images, rendered_images, masks)
            loss_landmark, landmarks_2d = landmark_loss(landmarks_3d, landmarks)
            loss_perceptual = perceptual_loss(images, rendered_images, masks)
            loss_coeff_reg = coefficient_regularization_loss(coeff)
            total_loss = 1.9 * loss_photometric + \
                         1.6e-3 * loss_landmark + \
                         0.2 * loss_perceptual + \
                         3e-4 * loss_coeff_reg
            # clean up
            total_loss.backward()
            optimizer.step()
            # info output
            if iteration % print_iter == 0:
                # print and save images
                print(("[Training] Epoch: %d, Iteration: %d: Total_loss=%0.6f " +
                      "L_photo=%0.6f L_landmark=%0.6f L_perc=%0.6f L_coeff_reg=%0.6f") %
                      (epoch, iteration,
                       total_loss.detach().cpu().numpy().item(),
                       loss_photometric.detach().cpu().numpy().item(),
                       loss_landmark.detach().cpu().numpy().item(),
                       loss_perceptual.detach().cpu().numpy().item(),
                       loss_coeff_reg.detach().cpu().numpy().item()))
                image = save_images(images, rendered_images, landmarks, landmarks_2d, epoch, iteration, vis_dir)
                # tensorboard
                global_step = train_length * epoch + iteration
                writer.add_scalar("train/total_loss", total_loss, global_step)
                writer.add_scalar("train/loss_photometric", loss_photometric, global_step)
                writer.add_scalar("train/loss_landmark", loss_landmark, global_step)
                writer.add_scalar("train/loss_perceptual", loss_perceptual, global_step)
                writer.add_scalar("train/loss_coeff_reg", loss_coeff_reg, global_step)
                writer.add_image("train/visualization", image, global_step, dataformats="HWC")
            if iteration % val_iter == 0:
                # val
                encoder.eval()
                decoder.eval()
                renderer.eval()
                for local_val_iteration, (images, landmarks, masks) in enumerate(val_loader):
                    images = images.cuda()
                    landmarks = landmarks.cuda()
                    masks = masks.cuda()
                    # forward
                    coeff = encoder(images)
                    face_projection, face_color, landmarks_3d, tri = decoder(coeff)
                    rendered_images, _ = renderer(face_projection, face_color, tri)
                    # loss
                    loss_photometric = photometric_loss(images, rendered_images, masks)
                    loss_landmark, landmarks_2d = landmark_loss(landmarks_3d, landmarks)
                    loss_perceptual = perceptual_loss(images, rendered_images, masks)
                    loss_coeff_reg = coefficient_regularization_loss(coeff)
                    total_loss = 1.9 * loss_photometric + \
                                 1.6e-3 * loss_landmark + \
                                 0.2 * loss_perceptual + \
                                 3e-4 * loss_coeff_reg
                    # print and save images
                    print(("[Validation] Iteration: %d: Total_loss=%0.6f " +
                          "L_photo=%0.6f L_landmark=%0.6f L_perc=%0.6f L_coeff_reg=%0.6f") %
                          (global_val_iteration,
                           total_loss.detach().cpu().numpy().item(),
                           loss_photometric.detach().cpu().numpy().item(),
                           loss_landmark.detach().cpu().numpy().item(),
                           loss_perceptual.detach().cpu().numpy().item(),
                           loss_coeff_reg.detach().cpu().numpy().item()))
                    image = save_images(images, rendered_images, landmarks, landmarks_2d,
                                        None, global_val_iteration, vis_dir)
                    # tensorboard
                    writer.add_scalar("val/total_loss", total_loss, global_val_iteration)
                    writer.add_scalar("val/loss_photometric", loss_photometric, global_val_iteration)
                    writer.add_scalar("val/loss_landmark", loss_landmark, global_val_iteration)
                    writer.add_scalar("val/loss_perceptual", loss_perceptual, global_val_iteration)
                    writer.add_scalar("val/loss_coeff_reg", loss_coeff_reg, global_val_iteration)
                    writer.add_image("val/visualization", image, global_val_iteration, dataformats="HWC")
                    global_val_iteration += 1
                    if local_val_iteration > 8:
                        break

        torch.save(encoder.state_dict(), save_encoder_path % epoch)
        if decoder_trainable:
            torch.save(decoder.state_dict(), save_decoder_path % epoch)


if __name__ == '__main__':
    main()
