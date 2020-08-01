from os import listdir
from os.path import isfile, join
import cv2


def custom_test_image():
    testpath = './test'
    onlyfiles = [f for f in listdir(testpath) if isfile(join(testpath, f))]
    for n in range(0, len(onlyfiles)):
        image = cv2.imread(join(testpath, onlyfiles[n]))
        dim = (28, 28)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_file = 'deploy/custom_images/' + onlyfiles[n]
        cv2.imwrite(img_file, img)


if __name__ == "__main__":
    custom_test_image()
