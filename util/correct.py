import os

if __name__ == '__main__':
    for file in os.listdir('.'):
        if file[-4:] == '.jpg':
            name = file[:-3]
            print(name)
