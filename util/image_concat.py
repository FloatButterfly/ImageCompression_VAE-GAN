from PIL import Image

UNIT_SIZE = 256
Target_size = 2 * UNIT_SIZE
save_path = '../datasets/edges2handbags/train_x4/'


def stitch():
    for i in range(0, 200):
        I1 = Image.open('./images/input_%03d_ground truth.png' % i)
        I2 = Image.open('./upsamle_x4_qp35_dbpn_binary/edges_%03d.bmp' % (i + 1))
        target = Image.new('RGB', (512, 256))
        target.paste(I2, (0, 0, 256, 256))
        target.paste(I1, (256, 0, 512, 256))
        target.save(save_path + '%d_AB.jpg' % i)


def stitch2():
    for i in range(1, 138561):
        I1 = Image.open('../datasets/edges2handbags/handbags/image_%d.jpg' % i)
        I2 = Image.open('../datasets/edges2handbags/dec_edgelr_256/edge_%d.jpg' % i)
        target = Image.new('RGB', (512, 256))
        target.paste(I2, (0, 0, 256, 256))
        target.paste(I1, (256, 0, 512, 256))
        target.save(save_path + '%d_AB.jpg' % i)


def cutedge():
    for i in range(1, 138568):
        I1 = Image.open('../datasets/edges2handbags/train/%d_AB.jpg' % i)
        target = Image.new('RGB', (256, 256))
        target = I1.crop((0, 0, 256, 256))
        target.save(save_path + 'edge_%d.jpg' % i)


def cutImage():
    for i in range(1, 138568):
        I1 = Image.open('../datasets/edges2handbags/train/%d_AB.jpg' % i)
        target = Image.new('RGB', (256, 256))
        target = I1.crop((256, 0, 512, 256))
        target.save(save_path + 'image_%d.jpg' % i)


if __name__ == '__main__':
    stitch2()
    # cutImage()
