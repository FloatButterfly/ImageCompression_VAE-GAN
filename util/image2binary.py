import numpy as np
from PIL import Image


def binary():
    img = np.array(Image.open('input_001_input.jpg').convert('L'))
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if img[i, j] <= 128:
                img[i, j] = 0
            else:
                img[i, j] = 255
    im = Image.fromarray(img)
    im.save('11.bmp')


def stitch():
    # for i in range(0, 200):
    I1 = Image.open('input_000_ground truth.png')
    I2 = Image.open('11.bmp')
    target = Image.new('RGB', (512, 256))
    target.paste(I2, (0, 0, 256, 256))
    target.paste(I1, (256, 0, 512, 256))
    target.save('11_AB.jpg')


if __name__ == '__main__':
    # binary()
    stitch()
