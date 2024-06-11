import os
import torch
import torchvision
import PIL.Image
import numpy as np

from deep_3d_face import (
    get_encoder,
    Decoder,
    Renderer
)

# settings
image_dir = "assets/images"
results_dir = "assets/results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
bfm_params_dir = "params/bfm"
encoder_params = "ckpts/encoder_latest.pth"
decoder_params = ""
decoder_trainable = False

def load_models(bfm_params_dir, encoder_params, decoder_params, decoder_trainable):
    encoder = get_encoder()
    encoder.load_state_dict(torch.load(encoder_params))
    encoder = encoder.cuda()
    decoder = Decoder(bfm_params_dir, decoder_trainable)
    if decoder_trainable and decoder_params:
        decoder.load_state_dict(torch.load(decoder_params))
    decoder = decoder.cuda()
    renderer = Renderer()
    renderer = renderer.cuda()
    return encoder, decoder, renderer

transforms = torchvision.transforms.Compose([
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5])
             ])

def main():
    # models
    encoder, decoder, renderer = load_models(bfm_params_dir, encoder_params, decoder_params, decoder_trainable)
    encoder.eval()
    decoder.eval()
    renderer.eval()
    # predict
    for frame in os.listdir(image_dir):
        # data preprocessing
        frame_path = os.path.join(image_dir, frame)
        pil_image = PIL.Image.open(frame_path).convert("RGB")
        batch_data = transforms(pil_image)
        batch_data = torch.reshape(batch_data, (1, 3, 256, 256))
        batch_data = batch_data.cuda()
        # forward passes
        with torch.no_grad():
            coeff = encoder(batch_data)
            face_projection, face_color, landmarks_3d, tri = decoder(coeff)
            # save the renderer image
            rendered_images, masks = renderer(face_projection, face_color, tri)
            remapped_images = (rendered_images[0] * 127.5 + 127.5) * masks[0]
            result = remapped_images.detach().cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)  # RGB image [0, 255] (h, w, c)

        pil_image = PIL.Image.fromarray(result)
        pil_image.save(os.path.join(results_dir, frame))


if __name__ == '__main__':
    main()
