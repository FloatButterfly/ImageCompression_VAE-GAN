from PIL import Image, ImageEnhance


def downsample():
    for i in range(0, 200):
        img = Image.open('./hbimages/input_%03d_input.jpg' % i)
        img_resized_x4 = img.resize((64, 64), Image.BICUBIC)
        # img_resized_x8 = img.resize((32, 32), Image.BICUBIC)
        img_resized_x4.save('./hbedge_x4/edge_%d.jpg' % i)
        # img_resized_x8.save('../datasets/edges2shoes/edgelr_x8/edge_%d.jpg' % i)


def upsample():
    for i in range(1, 201):
        img = Image.open('./dec_edge64_qp35/edges_%03d.jpg' % i)
        img_resized = img.resize((256, 256), Image.BILINEAR)
        img_resized.save('./upsample_x4_qp35/edges_%03d.jpg' % i)


def enhance():
    img = Image.open("./upsample/edges_001_qp30.jpg")
    # enhancer1 = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    enhancer1 = ImageEnhance.Sharpness(img).enhance(5.0)
    enhancer1.save("test1.jpg")


if __name__ == '__main__':
    # enhance()
    # upsample()
    downsample()
