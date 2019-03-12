import os


def main():
    root = r"."
    for item in os.listdir(root):
        if item[-4:] == ".png":
            file_x = os.path.join(root, item)
            file_y = os.path.join(root, item[:-4]+".jpg")
            os.rename(file_x, file_y)


if __name__ == "__main__":
    main()
