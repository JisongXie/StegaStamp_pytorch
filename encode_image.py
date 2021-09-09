import os
import glob
import bchlib
import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import transforms

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=r'./images')
    parser.add_argument('--secret', type=str, default='Stega!!')
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    encoder = torch.load(args.model)
    encoder.eval()
    if args.cuda:
        encoder = encoder.cuda()

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' ' * (7 - len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])
    secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0)
    if args.cuda:
        secret = secret.cuda()

    width = 400
    height = 400
    size = (width, height)
    to_tensor = transforms.ToTensor()

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        with torch.no_grad():
            for filename in files_list:
                image = Image.open(filename).convert("RGB")
                image = ImageOps.fit(image, size)
                image = to_tensor(image).unsqueeze(0)
                if args.cuda:
                    image = image.cuda()

                residual = encoder((secret, image))
                encoded = image + residual
                if args.cuda:
                    residual = residual.cpu()
                    encoded = encoded.cpu()
                encoded = np.array(encoded.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

                residual = residual[0] + .5
                residual = np.array(residual.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

                save_name = os.path.basename(filename).split('.')[0]

                im = Image.fromarray(encoded)
                im.save(args.save_dir + '/' + save_name + '_hidden.png')

                im = Image.fromarray(residual)
                im.save(args.save_dir + '/' + save_name + '_residual.png')


if __name__ == "__main__":
    main()
