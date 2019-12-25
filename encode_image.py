
import os
import glob
import bchlib
import numpy as np
from PIL import Image,ImageOps

import torch

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--secret', type=str, default='Stega!!')
    parser.add_argument('--secret_size', type=int, default=100)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    encoder = torch.load(args.model).cuda()

    width = 400
    height = 400

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0,0,0,0])

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)
        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = np.array(ImageOps.fit(image, size), dtype=np.float32)
            image /= 255.

            hidden_img, residual = encoder(secret, image)
            
            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            # raw_img = (image * 255).astype(np.uint8)
            residual = residual[0]+.5

            residual = (residual * 255).astype(np.uint8)

            save_name = filename.split('/')[-1].split('.')[0]

            im = Image.fromarray(np.array(rescaled))
            im.save(args.save_dir + '/'+save_name+'_hidden.png')

            im = Image.fromarray(np.squeeze(np.array(residual)))
            im.save(args.save_dir + '/'+save_name+'_residual.png')

if __name__ == "__main__":
    main()