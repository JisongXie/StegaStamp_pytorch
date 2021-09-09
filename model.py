import sys

sys.path.append("PerceptualSimilarity\\")
import os
import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import torch.nn.functional as F
from torchvision import transforms


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class StegaStampEncoder(nn.Module):
    def __init__(self):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = Dense(100, 7500, activation='relu', kernel_initializer='he_normal')

        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

    def forward(self, inputs):
        secrect, image = inputs
        secrect = secrect - .5
        image = image - .5

        secrect = self.secret_dense(secrect)
        secrect = secrect.reshape(-1, 3, 50, 50)
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect)

        inputs = torch.cat([secrect_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        self.localization = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(320000, 128, activation='relu'),
            nn.Linear(128, 6)
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, image):
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        return transformed_image


class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size=100):
        super(StegaStampDecoder, self).__init__()
        self.secret_size = secret_size
        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(21632, 512, activation='relu'),
            Dense(512, secret_size, activation=None))

    def forward(self, image):
        image = image - .5
        transformed_image = self.stn(image)
        return torch.sigmoid(self.decoder(transformed_image))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2D(3, 8, 3, strides=2, activation='relu'),
            Conv2D(8, 16, 3, strides=2, activation='relu'),
            Conv2D(16, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 1, 3, activation=None))

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output, x


def transform_net(encoded_image, args, global_step):
    sh = encoded_image.size()
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, args.batch_size)  # [batch_size, 3, 1, 1]
    jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
    rnd_noise = torch.rand(1)[0] * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # blur
    N_blur = 7
    f = utils.random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                 wmin_line=3)
    if args.cuda:
        f = f.cuda()
    encoded_image = F.conv2d(encoded_image, f, bias=None, padding=int((N_blur - 1) / 2))

    # noise
    noise = torch.normal(mean=0, std=rnd_noise, size=encoded_image.size(), dtype=torch.float32)
    if args.cuda:
        noise = noise.cuda()
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # contrast & brightness
    contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0], contrast_params[1])
    contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
    if args.cuda:
        contrast_scale = contrast_scale.cuda()
        rnd_brightness = rnd_brightness.cuda()
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # saturation
    sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1)
    if args.cuda:
        sat_weight = sat_weight.cuda()
    encoded_image_lum = torch.mean(encoded_image * sat_weight, dim=1).unsqueeze_(1)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # jpeg
    encoded_image = encoded_image.reshape([-1, 3, 400, 400])
    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0,
                                                       quality=jpeg_quality)

    return encoded_image


def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc


def build_model(encoder, decoder, discriminator, lpips_fn, secret_input, image_input, l2_edge_gain,
                borders, secret_size, M, loss_scales, yuv_scales, args, global_step, writer):
    test_transform = transform_net(image_input, args, global_step)

    input_warped = torchgeometry.warp_perspective(image_input, M[:, 1, :, :], dsize=(400, 400), flags='bilinear')
    mask_warped = torchgeometry.warp_perspective(torch.ones_like(input_warped), M[:, 1, :, :], dsize=(400, 400),
                                                 flags='bilinear')
    input_warped += (1 - mask_warped) * image_input

    residual_warped = encoder((secret_input, input_warped))
    encoded_warped = residual_warped + input_warped

    residual = torchgeometry.warp_perspective(residual_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')

    if borders == 'no_edge':
        encoded_image = image_input + residual
    elif borders == 'black':
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        input_unwarped = torchgeometry.warp_perspective(image_input, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
    elif borders.startswith('random'):
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400),
                                              flags='bilinear')
        encoded_image = residual_warped + input_unwarped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        input_unwarped = torchgeometry.warp_perspective(input_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        ch = 3 if borders.endswith('rgb') else 1
        encoded_image += (1 - mask) * torch.ones_like(residual) * torch.rand([ch])
    elif borders == 'white':
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400),
                                              flags='bilinear')
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        input_unwarped = torchgeometry.warp_perspective(input_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        encoded_image += (1 - mask) * torch.ones_like(residual)
    elif borders == 'image':
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400),
                                              flags='bilinear')
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        encoded_image += (1 - mask) * torch.roll(image_input, 1, 0)

    if borders == 'no_edge':
        D_output_real, _ = discriminator(image_input)
        D_output_fake, D_heatmap = discriminator(encoded_image)
    else:
        D_output_real, _ = discriminator(input_warped)
        D_output_fake, D_heatmap = discriminator(encoded_warped)

    transformed_image = transform_net(encoded_image, args, global_step)
    decoded_secret = decoder(transformed_image)
    bit_acc, str_acc = get_secret_acc(secret_input, decoded_secret)

    normalized_input = image_input * 2 - 1
    normalized_encoded = encoded_image * 2 - 1
    lpips_loss = torch.mean(lpips_fn(normalized_input, normalized_encoded))

    cross_entropy = nn.BCELoss()
    if args.cuda:
        cross_entropy = cross_entropy.cuda()
    secret_loss = cross_entropy(decoded_secret, secret_input)

    size = (int(image_input.shape[2]), int(image_input.shape[3]))
    gain = 10
    falloff_speed = 4
    falloff_im = np.ones(size)
    for i in range(int(falloff_im.shape[0] / falloff_speed)):  # for i in range 100
        falloff_im[-i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
        falloff_im[i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
    for j in range(int(falloff_im.shape[1] / falloff_speed)):
        falloff_im[:, -j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
        falloff_im[:, j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
    falloff_im = 1 - falloff_im
    falloff_im = torch.from_numpy(falloff_im).float()
    if args.cuda:
        falloff_im = falloff_im.cuda()
    falloff_im *= l2_edge_gain

    encoded_image_yuv = color.rgb_to_yuv(encoded_image)
    image_input_yuv = color.rgb_to_yuv(image_input)
    im_diff = encoded_image_yuv - image_input_yuv
    im_diff += im_diff * falloff_im.unsqueeze_(0)
    yuv_loss = torch.mean((im_diff) ** 2, axis=[0, 2, 3])
    yuv_scales = torch.Tensor(yuv_scales)
    if args.cuda:
        yuv_scales = yuv_scales.cuda()
    image_loss = torch.dot(yuv_loss, yuv_scales)

    D_loss = D_output_real - D_output_fake
    G_loss = D_output_fake
    loss = loss_scales[0] * image_loss + loss_scales[1] * lpips_loss + loss_scales[2] * secret_loss
    if not args.no_gan:
        loss += loss_scales[3] * G_loss

    writer.add_scalar('loss/image_loss', image_loss, global_step)
    writer.add_scalar('loss/lpips_loss', lpips_loss, global_step)
    writer.add_scalar('loss/secret_loss', secret_loss, global_step)
    writer.add_scalar('loss/G_loss', G_loss, global_step)
    writer.add_scalar('loss/loss', loss, global_step)

    writer.add_scalar('metric/bit_acc', bit_acc, global_step)
    writer.add_scalar('metric/str_acc', str_acc, global_step)
    if global_step % 20 == 0:
        writer.add_image('input/image_input', image_input[0], global_step)
        writer.add_image('input/image_warped', input_warped[0], global_step)
        writer.add_image('encoded/encoded_warped', encoded_warped[0], global_step)
        writer.add_image('encoded/residual_warped', residual_warped[0] + 0.5, global_step)
        writer.add_image('encoded/encoded_image', encoded_image[0], global_step)
        writer.add_image('transformed/transformed_image', transformed_image[0], global_step)
        writer.add_image('transformed/test', test_transform[0], global_step)

    return loss, secret_loss, D_loss, bit_acc, str_acc
