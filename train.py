import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

from model.unet import UNet
import numpy as np

import logging
from data import get_data_iter

import copy
import os
from tqdm import trange
from diffusion import GaussianDiffusionTrainer, DDPM_Sampler, DDIM_Sampler
from utils import restore_checkpoint, save_checkpoint, EMAHelper, ConditionalEmbedding
from evaluate import sample, evaluate


def check_data(data_cycle, FLAGS):
    samples = next(data_cycle) # with label
    images = make_grid(samples, nrow=8)
    save_image(images, os.path.join(FLAGS.img_dir, 'real_{}.png'.format(FLAGS.config.data.dataset)))


def train(FLAGS):

    config = FLAGS.config

    # model initialization
    net_model = UNet(
                    T=config.training.T, 
                    ch=config.model.ch, 
                    ch_mult=config.model.ch_mult, 
                    attn=config.model.attn,
                    num_res_blocks=config.model.num_res_blocks, 
                    dropout=config.model.dropout)
    
    # label embedding
    cemblayer = ConditionalEmbedding(10, config.data.num_classes, 
                                     config.data.num_classes).to(FLAGS.device)

    # optimizer
    optim = torch.optim.Adam(net_model.parameters(), lr=config.training.lr)
    # warmup
    warmup_lr = lambda step:  min(step, config.training.warmup) / config.training.warmup
    # scheduler
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    # DDPM trainer
    trainer = GaussianDiffusionTrainer(net_model, config).to(FLAGS.device)

    # setup ema
    ema_helper = EMAHelper(mu=config.training.ema_decay)
    ema_helper.register(net_model)
    
    if config.evaluate.sampler == 'ddpm':
        net_sampler = DDPM_Sampler(net_model, config).to(FLAGS.device)
    elif config.evaluate.sampler == 'ddim':
        net_sampler = DDIM_Sampler(net_model, config).to(FLAGS.device) # not working well
    else:
        raise NotImplementedError

    # setup tensorboard
    writer = SummaryWriter(FLAGS.logging_dir)
   
    # backup all arguments
    with open(os.path.join(FLAGS.logging_dir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    start_step = 0
    # restore checkpoint
    if FLAGS.restore_dir is not None:
        logging.info('loading checkpoint from %s' % FLAGS.restore_dir)
        state = restore_checkpoint(FLAGS.restore_dir)
        net_model.load_state_dict(state['net_model'])
        ema_helper.load_state_dict(state['ema_model'])
        sched.load_state_dict(state['sched'])
        optim.load_state_dict(state['optim'])
        cemblayer.load_state_dict(state['cemblayer'])
        start_step = state['step']

    # build data iterator
    data_iter = get_data_iter(FLAGS.config)
    check_data(data_iter, FLAGS) # sample real images and save

    # start training
    with trange(start_step, config.training.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0, lab = next(data_iter).to(FLAGS.device)

            lab = lab.to(FLAGS.device)
            cemb = cemblayer(lab)
            cemb[np.where(np.random.rand(x_0.shape[0])<config.training.threshold)] = 0

            loss = trainer(x_0, cemb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), config.training.grad_clip)
            optim.step()
            sched.step()
            
            # ema update
            ema_helper.update(net_model)
    
            # log
            logging.info('step: %d, loss: %.3f' % (step, loss))
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # save checkpoint
            if config.evaluate.save_step > 0 and step % config.evaluate.save_step == 0:
                state = {'net_model':net_model.state_dict(), 
                        'ema_model':ema_helper.state_dict(),
                        'cemblayer':cemblayer.module.state_dict(), 
                        'sched':sched.state_dict(), 
                        'optim':optim.state_dict(), 
                        'step':step}
                save_checkpoint(FLAGS.logging_dir, state)

            # sample
            if config.evaluate.sample_step > 0 and step % config.evaluate.sample_step == 0:
                sample(FLAGS, net_model, cemblayer, net_sampler, writer, step)
                
                

          
            # evaluate
            # if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
            #     evaluate(FLAGS, net_model, net_sampler, ema_helper, writer, step)
            #     net_IS, net_FID, _ = evaluate(net_sampler, net_model)
            #     ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
            #     metrics = {
            #         'IS': net_IS[0],
            #         'IS_std': net_IS[1],
            #         'FID': net_FID,
            #         'IS_EMA': ema_IS[0],
            #         'IS_std_EMA': ema_IS[1],
            #         'FID_EMA': ema_FID
            #     }
            #     pbar.write(
            #         "%d/%d " % (step, FLAGS.total_steps) +
            #         ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
            #     for name, value in metrics.items():
            #         writer.add_scalar(name, value, step)
            #     writer.flush()
            #     with open(os.path.join(FLAGS.loggingdir, 'eval.txt'), 'a') as f:
            #         metrics['step'] = step
            #         f.write(json.dumps(metrics) + "\n")
    writer.close()

