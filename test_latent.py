import os
from itertools import islice

import numpy as np
import torch

from data import CreateDataLoader
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1  # test code only supports num_threads=1
    opt.batch_size = 1  # test code only supports batch_size=1
    opt.serial_batches = True  # no shuffle

    # create dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    print('Loading model %s' % opt.model)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
    webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

    # sample random z
    z_log_dir = os.path.join(opt.results_dir, 'z_vector.txt')

    # test stage
    for x in range(0, 8):
        z_encoded = torch.zeros([1, 8])
        a = np.linspace(-3, 3, 20)
        orignal_code = z_encoded.clone()
        with open(z_log_dir, "a") as log_file:
            np.savetxt(log_file, z_encoded, fmt='%.4f')
            for i, data in enumerate(islice(dataset, opt.num_test)):
                model.set_input(data)
                print('process input image %3.3d/%3.3d' % (i, opt.num_test))
                z_encoded = orignal_code.clone()
                z0 = torch.tensor(z_encoded, dtype=torch.float32)
                real_A, fake_B, real_B = model.test(z0, encode=False)
                images = [real_A, real_B, fake_B]
                names = ['input', 'ground truth', 'encoded']
                for j in range(0, 15):
                    z_encoded[0][x] = a[j]
                    with open(z_log_dir, "a") as log_file:
                        np.savetxt(log_file, z_encoded, fmt='%.4f')
                    encoded = torch.tensor(z_encoded, dtype=torch.float32, device='cuda:0', requires_grad=True)
                    print('encoded: ', encoded)
                    real_A, fake_B, real_B = model.test(encoded, encode=False)
                    print('x:', x, ' i:', i)
                    images.append(fake_B)
                    names.append('random_variable_%d' % j)
                img_path = 'input_%3.3d_%d' % (i, x)
                save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.fineSize)

    webpage.save()
