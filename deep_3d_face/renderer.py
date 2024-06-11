import math
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)


class CameraParams():
    def __init__(self):
        self.camera_z_pos = 10
        self.image_size = 256
        self.focal_length = 1015
        self.znear = 0.01
        self.zfar = 50


camera_params = CameraParams()
device = torch.device("cuda:0")


class Renderer(torch.nn.Module):
    def __init__(self):
        super(Renderer, self).__init__()
        R, T = look_at_view_transform(eye=((0, 0, camera_params.camera_z_pos),),
                                      up=((0, 1, 0),),
                                      at=((0, 0, 0),))
        fov = 2 * math.atan(camera_params.image_size / 2.0 / camera_params.focal_length) * 180.0 / math.pi
        cameras = FoVPerspectiveCameras(
            znear=camera_params.znear,
            zfar=camera_params.zfar,
            aspect_ratio=1.0,
            fov=fov,
            degrees=True, 
            R=R, 
            T=T,
            device=device
        )
        lights = DirectionalLights(ambient_color=((1, 1, 1),),
                                   diffuse_color=((0, 0, 0),),
                                   specular_color=((0, 0, 0),),
                                   device=device)
        raster_settings = RasterizationSettings(
            image_size=camera_params.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(cameras=cameras, lights=lights, device=device)
        )

    def forward(self, face_projection, face_color, triangles):
        """ parameters are exactly the outputs of the decoder
        :param face_projection:
        :param face_color:
        :param triangles:
        :return: a rendered image, RGB, (-1, 1), (bs, 3, h, w)
        """
        bs = face_projection.shape[0]
        mesh = Meshes(face_projection,
                      triangles.reshape((1, -1, 3)).repeat(bs, 1, 1),
                      TexturesVertex(face_color))
        rendered_image = self.renderer(mesh)
        rendered_image = rendered_image[:, :, :, :3]  # (bs, h, w, c) (0, 1) RGB
        mask = self._cal_mask(rendered_image)
        rendered_image = rendered_image.permute(0, 3, 1, 2) * 2.0 - 1.0
        return rendered_image, mask

    def _cal_mask(self, renderer_image):
        """
        :param renderer_image: (torch.tensor) (bs, h, w, c) (0, 1) RGB
        :return: mask (bs, 3, h, w), 1-> face area
        """
        images = renderer_image.clone().detach()
        images = images.permute(0, 3, 1, 2) * 255  # (bs, 3, h, w) (0, 255)
        images = images.long()
        tensor = torch.sum(images, dim=1)
        another_tensor = torch.ones_like(tensor) * 255 * 3  # white background
        #another_tensor = torch.zeros_like(tensor)  # black background
        mask = torch.eq(tensor, another_tensor).long().float()
        mask = 1.0 - mask  # 1-> face area
        mask = torch.unsqueeze(mask, 1).repeat(1, 3, 1, 1)
        return mask
