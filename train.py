import os
import yaml
import random
import model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
from dataset import StegaData
from torch.utils.data import DataLoader

with open('cfg\setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

def main():

    dataset = StegaData(args.train_path, args.secret_size, size=(400, 400))
    dataloader = DataLoader(datase, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    encoder = model.StegaStampEncoder()
    decoder = model.StegaStampDecoder(args.secret_size=100)
    discriminator = model.Discriminator()
    model.build_model()

    d_vars = discriminator.parameters()
    g_vars = [encoder.parameters(), decoder.parameters()]
 
    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    total_steps = len(StegaData) // args.batch_size + 1
    global_step = 0

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            no_im_loss = global_step < args.no_im_loss_steps
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp, args.secret_loss_scale)
            G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step-args.l2_edge_delay) / args.l2_edge_ramp, args.l2_edge_gain)
            
            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran
            
            global_step += 1
            dst, rect = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
            
            loss, secret_loss, D_loss, bit_acc = build_model(encoder, decoder, discriminator, secret_input, image_input, l2_edge_gain, 
                    borders, secret_size, dst, rect, loss_scales, yuv_scales, args, global_step)
            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()
            
            if global_step % 10 == 0:
                print('Loss = {:.4f}'.format(loss))
            