import os
import torch
import numpy as np


class Decoder(torch.nn.Module):
    def __init__(self, bfm_params_dir, trainable=False):
        super(Decoder, self).__init__()
        self.trainable = trainable
        self._load_face_model(bfm_params_dir)

    def forward(self, coeff):
        """
        :param coeff: coefficients predicted by the encoder
        :return:
        """
        id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self._spilt_coeff(coeff)
        face_shape = self._cal_shape(id_coeff, ex_coeff)
        face_texture = self._cal_texture(tex_coeff)
        normals = self._cal_normals(face_shape)
        rotation = self._cal_rotation_matrix(angles)
        face_color, _ = self._cal_color_with_illumination(face_texture, normals, gamma)
        face_projection = face_shape.bmm(rotation) + translation.view(-1, 1, 3)
        landmarks_3d = face_projection[:, self.keypoints, :]
        tri = self.tri
        return face_projection, face_color, landmarks_3d, self.tri_device

    def _load_face_model(self, face_model_dir):
        meanshape = np.load(os.path.join(face_model_dir, "meanshape.npy")).astype("float32")
        idBase = np.load(os.path.join(face_model_dir, "idBase.npy")).astype("float32")
        exBase = np.load(os.path.join(face_model_dir, "exBase.npy")).astype("float32")
        meantex = np.load(os.path.join(face_model_dir, "meantex.npy")).astype("float32")
        texBase = np.load(os.path.join(face_model_dir, "texBase.npy")).astype("float32")
        self.tri = np.load(os.path.join(face_model_dir, "tri.npy")).astype("int64") - 1
        self.keypoints = np.load(os.path.join(face_model_dir, "keypoints.npy")).squeeze().astype("int64") - 1
        self.point_buf = np.load(os.path.join(face_model_dir, "point_buf.npy")).astype("int64") - 1
        # to torch
        if self.trainable:
            self.meanshape = torch.nn.Parameter(torch.from_numpy(meanshape))
            self.idBase = torch.nn.Parameter(torch.from_numpy(idBase))
            self.exBase = torch.nn.Parameter(torch.from_numpy(exBase))
            self.meantex = torch.nn.Parameter(torch.from_numpy(meantex))
            self.texBase = torch.nn.Parameter(torch.from_numpy(texBase))
            self.register_buffer("tri_device", torch.from_numpy(self.tri))
        else:
            self.register_buffer("meanshape", torch.from_numpy(meanshape))
            self.register_buffer("idBase", torch.from_numpy(idBase))
            self.register_buffer("exBase", torch.from_numpy(exBase))
            self.register_buffer("meantex", torch.from_numpy(meantex))
            self.register_buffer("texBase", torch.from_numpy(texBase))
            self.register_buffer("tri_device", torch.from_numpy(self.tri))

    def _cal_shape(self, id_coeff, ex_coeff):
        bs = id_coeff.shape[0]
        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
                     torch.einsum('ij,aj->ai', self.exBase, ex_coeff) + \
                     self.meanshape
        face_shape = face_shape.view(bs, -1, 3)
        face_shape = face_shape - self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)
        return face_shape

    def _cal_texture(self, tex_coeff):
        bs = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + \
                       self.meantex
        face_texture = face_texture.view(bs, -1, 3)
        return face_texture

    def _cal_normals(self, face_shape):
        v1 = face_shape[:, self.tri[:, 0], :]
        v2 = face_shape[:, self.tri[:, 1], :]
        v3 = face_shape[:, self.tri[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2, 2)
        empty = torch.zeros((face_norm.size(0), 1, 3),
                            dtype=face_norm.dtype,
                            device=face_norm.device)
        face_norm = torch.cat((face_norm, empty), 1)
        v_norm = face_norm[:, self.point_buf, :].sum(2)
        v_norm = v_norm / v_norm.norm(dim=2).unsqueeze(2)
        return v_norm

    @staticmethod
    def _cal_rotation_matrix(angles):
        bs = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])
        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(bs * 3, 1, 1).view(3, bs, 3, 3)
        rotXYZ = rotXYZ.to(angles.device)
        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz
        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])
        return rotation.permute(0, 2, 1)

    @staticmethod
    def _cal_color_with_illumination(face_texture, norm, gamma):
        n_b, num_vertex, _ = face_texture.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8
        gamma = gamma.permute(0, 2, 1)
        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)
        Y0 = torch.ones(n_v_full).float() * a0 * c0
        if gamma.is_cuda:
            Y0 = Y0.cuda()
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []
        arrH.append(Y0)
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))
        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)
        face_color = face_texture * lighting
        face_color = torch.clamp(face_color / 255, 0.0, 1.0)
        return face_color, lighting

    @staticmethod
    def _spilt_coeff(coeff):
        id_coeff = coeff[:, 0:80]
        ex_coeff = coeff[:, 80:144]
        tex_coeff = coeff[:, 144:224]
        angles = coeff[:, 224:227]
        gamma = coeff[:, 227:254]
        translation = coeff[:, 254:257]
        return id_coeff, ex_coeff, tex_coeff, angles, gamma, translation
