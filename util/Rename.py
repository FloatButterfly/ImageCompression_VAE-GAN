import os


def main():
    for file in os.listdir('.'):
        new_name = file + '.jpg'
        os.rename(file, new_name)


def rename():
    for i, filename in enumerate(os.listdir('../../datasets/train_together')):
        if i <= 3:
            print(filename)


if __name__ == '__main__':
    # main()
    rename()
