import torch

from .identity_encoder import Backbone as IdentityEncoder
from .renderer import camera_params


class PhotometricLoss(torch.nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, original_images, rendered_images, masks):
        loss = torch.nn.functional.mse_loss(rendered_images * masks,
                                            original_images * masks,
                                            reduction="mean")
        return loss


class LandmarkLoss(torch.nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        proj_mat = torch.tensor([[camera_params.focal_length, 0.0, camera_params.image_size / 2.0],
                                 [0.0, camera_params.focal_length, camera_params.image_size / 2.0],
                                 [0.0, 0.0, 1.0]])
        proj_mat = torch.reshape(proj_mat, (1, 3, 3))
        self.register_buffer("proj_mat", proj_mat.float())
        weight = torch.ones([1, 68, 2])
        weight[:, 27:36, :] = 20  # nose
        weight[:, 61:64, :] = 20  # inner mouth upper
        weight[:, 65:68, :] = 20  # inner mouth lower
        self.register_buffer("weight", weight.float())

    def forward(self, predicted_landmarks, target_landmarks):
        """
        :param predicted_landmarks: (batch_size, 68, 3)
        :param target_landmarks: (batch_size, 68, 2)
        :return:
        """
        bs = predicted_landmarks.shape[0]
        proj_mat = self.proj_mat.repeat((bs, 1, 1))
        predicted_landmarks[:, :, 2] = camera_params.camera_z_pos - predicted_landmarks[:, :, 2]
        aug_proj = predicted_landmarks.bmm(proj_mat.permute(0, 2, 1))
        face_proj = aug_proj[:, :, 0:2] / aug_proj[:, :, 2:]  # (bs, 68, 2)
        face_proj[:, :, 1] = camera_params.image_size - face_proj[:, :, 1]
        face_proj = torch.reshape(face_proj, (bs, -1))
        loss = torch.nn.functional.mse_loss(face_proj,
                                            target_landmarks,
                                            reduction="none")
        weight = self.weight.repeat((bs, 1, 1)).reshape((bs, -1))
        loss = torch.mean(loss * weight)
        return loss, torch.reshape(face_proj, (bs, 68, 2))


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        params_path = "params/model_ir_se50.pth"
        self.identity_encoder = IdentityEncoder(50, 0.4, "ir_se")
        self.identity_encoder.load_state_dict(torch.load(params_path))
        self.identity_encoder.eval()
        self.lower = int(camera_params.image_size / 16.0)
        self.upper = camera_params.image_size - self.lower

    def forward(self, original_images, rendered_images, masks):
        if masks is not None:
            original_images = original_images * masks
            rendered_images = rendered_images * masks
        ori = original_images[:, :, self.lower:self.upper, self.lower:self.upper]
        ren = rendered_images[:, :, self.lower:self.upper, self.lower:self.upper]
        ori = torch.nn.functional.interpolate(ori, [112, 112], mode="bilinear", align_corners=True).contiguous()
        ren = torch.nn.functional.interpolate(ren, [112, 112], mode="bilinear", align_corners=True).contiguous()
        ori_embedding = self.identity_encoder(ori)
        ren_embedding = self.identity_encoder(ren)
        loss = 1.0 - torch.mean(torch.cosine_similarity(ori_embedding, ren_embedding, dim=1))
        return loss



class CoefficientRegularizationLoss(torch.nn.Module):
    def __init__(self):
        super(CoefficientRegularizationLoss, self).__init__()
        self.w_id = 1.0
        self.w_ex = 0.8
        self.w_tex = 1.7e-3

    def forward(self, coeff):
        id_coeff = coeff[:, 0:80]
        ex_coeff = coeff[:, 80:144]
        tex_coeff = coeff[:, 144:224]
        loss = self.w_id * torch.mean(id_coeff * id_coeff) + \
               self.w_ex * torch.mean(ex_coeff * ex_coeff) + \
               self.w_tex * torch.mean(tex_coeff * tex_coeff)
        return loss
