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
    if opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    z_log_dir = os.path.join(opt.results_dir, 'z_vector.txt')
    std_log_dir = os.path.join(opt.results_dir, 'z_std.txt')
    logvar_log_dir = os.path.join(opt.results_dir, 'z_logvar.txt')
    # test stage
    for i, data in enumerate(islice(dataset, opt.num_test)):
        model.set_input(data)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test))
        encode = True
        z_encoded = model.z_encode()
        z_encoded = z_encoded.cpu().detach().numpy()
        orignal_code = z_encoded.copy()
        with open(z_log_dir, "a") as log_file:
            np.savetxt(log_file, z_encoded, fmt='%.4f')
        for x in range(0, 8):
            z_encoded = orignal_code.copy()
            real_A, fake_B, real_B = model.test(encode=encode)
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
            randnum = torch.rand(10).cpu()
            randnum, _ = torch.sort(randnum)
            center = z_encoded[0][x]
            # with original x as center
            # randnum = randnum * 10 + (center - 5)
            # in interval [-5,5]
            randnum = randnum * 10 - 5
            # print('computed randnum:', randnum)
            for j in range(0, 10):
                # print('to replace: ', z_encoded[0][x])
                z_encoded[0][x] = randnum[j]
                # print('after replace', z_encoded)
                with open(z_log_dir, "a") as log_file:
                    np.savetxt(log_file, z_encoded, fmt='%.4f')
                # z_encoded = torch.from_numpy(z_encoded)
                # print('test encoded: ', test_encoded)
                encoded = torch.tensor(z_encoded, device='cuda:0', requires_grad=True)
                # print('encoded: ', encoded)
                real_A, fake_B, real_B = model.test(encoded, encode=False)
                print('x:', x, ' i:', i)
                images.append(fake_B)
                names.append('random_variable_%d' % (j))
            img_path = 'input_%3.3d_%d' % (i, x)
            save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.fineSize)

        # for nn in range(opt.n_samples + 1):
        #     encode = nn == 0 and not opt.no_encode
        #     real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        #     if nn == 0:
        #         images = [real_A, real_B, fake_B]
        #         names = ['input', 'ground truth', 'encoded']
        #     else:
        #         images.append(fake_B)
        #         names.append('random_sample%2.2d' % nn)
    webpage.save()
