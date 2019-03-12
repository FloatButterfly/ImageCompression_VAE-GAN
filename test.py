import os
from itertools import islice

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
    # if opt.sync:
    #     z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    # sum_msssim = 0.0
    # sum_ssim = 0.0
    # sum_psnr = 0.0

    # psnr_log_dir = os.path.join(opt.results_dir, 'psnr.txt')
    # ssim_log_dir = os.path.join(opt.results_dir, 'ssim.txt')
    # msssim_log_dir = os.path.join(opt.results_dir, 'msssim.txt')
    z_log_dir = os.path.join(opt.results_dir, 'z_vector.txt')
    std_log_dir = os.path.join(opt.results_dir, 'z_std.txt')
    logvar_log_dir = os.path.join(opt.results_dir, 'z_logvar.txt')
    # test stage
    for i, data in enumerate(islice(dataset, opt.num_test)):
        model.set_input(data)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test))
        #     # if not opt.sync:
        #     z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
        encode = True
        z, real_A, fake_B, real_B, logvar = model.test(encode=encode)
        # print(z)
        # with open(z_log_dir, "a") as log_file:
        #     np.savetxt(log_file, z.cpu().detach().numpy())
        # with open(std_log_dir, "a") as log_file:
        #     np.savetxt(log_file, std.cpu().detach().numpy())
        # with open(logvar_log_dir, "a") as log_file:
        #     np.savetxt(log_file, logvar.cpu().detach().numpy())
        images = [real_A, real_B, fake_B]
        names = ['input', 'ground truth', 'encoded']
        # psnr
        # if opt.test_psnr:
        #     # m = pytorch_msssim.psnr(fake_B.cpu().numpy(), real_B.cpu().numpy())
        #     m = measure.compare_psnr(fake_B.cpu().numpy(), real_B.cpu().numpy())
        #     sum_psnr += m
        #     print(m)
        #     m = np.atleast_1d(m)
        #     with open(psnr_log_dir, "a") as log_file:
        #         np.savetxt(log_file, m)
        #
        # # ssim
        # if opt.test_ssim:
        #     # m = pytorch_msssim.ssim(fake_B, real_B)
        #     print(np.asarray(fake_B.cpu().numpy().shape))
        #     m = measure.compare_ssim(fake_B.cpu().numpy(), real_B.cpu().numpy(), multichannel=True, win_size=11)
        #     sum_ssim += m
        #     print(m)
        #     m = np.atleast_1d(m)
        #     with open(ssim_log_dir, "a") as log_file:
        #         np.savetxt(log_file, m)
        #
        # # mssim
        # if opt.test_msssim:
        #     m = pytorch_msssim.msssim(fake_B, real_B)
        #     sum_msssim += m
        #     print(m)
        #     m = np.atleast_1d(m)
        #     with open(msssim_log_dir, "a") as log_file:
        #         np.savetxt(log_file, m)

        # for nn in range(opt.n_samples + 1):
        #     encode = nn == 0 and not opt.no_encode
        #     real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        #     if nn == 0:
        #         images = [real_A, real_B, fake_B]
        #         names = ['input', 'ground truth', 'encoded']
        #     else:
        #         images.append(fake_B)
        #         names.append('random_sample%2.2d' % nn)

        img_path = 'input_%3.3d' % i
        save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.fineSize)
    # save the mean of the ssim,msssim,psnr
    # if opt.test_msssim:
    #     m_mean = sum_msssim / opt.num_test
    #     print(m_mean)
    #     with open(msssim_log_dir, "a") as log_file:
    #         log_file.write("mean msssim:   ")
    #         np.savetxt(log_file, np.atleast_1d(m_mean))
    #
    # if opt.test_ssim:
    #     m_mean = sum_ssim / opt.num_test
    #     print(m_mean)
    #     with open(ssim_log_dir, "a") as log_file:
    #         log_file.write("mean ssim:   ")
    #         np.savetxt(log_file, np.atleast_1d(m_mean))
    #
    # if opt.test_psnr:
    #     m_mean = sum_psnr / opt.num_test
    #     print(m_mean)
    #     with open(psnr_log_dir, "a") as log_file:
    #         log_file.write("mean psnr:   ")
    #         np.savetxt(log_file, np.atleast_1d(m_mean))
    webpage.save()
