import torch
from torchvision.utils import make_grid, save_image
from metrics import compute_metrics
import os

from data import get_data_iter

def sample(FLAGS, net_model, cemblayer, sampler, writer, step):

    config = FLAGS.config
    
    net_model.eval()
    cemblayer.eval()

    lab = torch.ones(config.evaluate.sample_size, dtype=torch.long) * \
        torch.arange(start = 0, end = config.data.num_classes).reshape(-1, 1)
    lab = lab.reshape(-1,1).squeeze().to(FLAGS.device)
    cemb = cemblayer(lab)

    x_T = torch.randn(config.evaluate.sample_size*config.data.num_classes, 
                      3, config.data.image_size, config.data.image_size)
    x_T = x_T.to(FLAGS.device)  

    with torch.no_grad():
        x_0 = sampler(x_T, cemb)
        grid = (make_grid(x_0, nrow=config.evaluate.sample_size) + 1) / 2
        path = os.path.join(
            FLAGS.img_dir, f'{step}_{config.evaluate.sampler}.png')
        save_image(grid, path)
        writer.add_image('sample', grid, step)

    net_model.train()
    cemblayer.train()
    return x_0



def evaluate(FLAGS):
    config = FLAGS.config

    # build eval data iterator
    data_iter = get_data_iter(config, train=False)
   
    # x_0 = sample(FLAGS, net_model, net_sampler, writer, step)

    pass


    