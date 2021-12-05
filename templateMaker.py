import cv2 as cv
import numpy as np

# Reads number images and converts them to binary templates


numberList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
numbersPath = "STMToWM_Medium/numbers/"
templatePath = "STMToWM_Medium/templates/"
resizePixels = 70


def main():
    # Work on images from numberList...
    for i in numberList:
        # Load reference image
        numImage = cv.imread(numbersPath + "num_" +
                             str(i) + ".jpg", cv.IMREAD_GRAYSCALE)
        # Resize image
        imgResized = cv.resize(numImage,
                               (resizePixels, resizePixels),
                               interpolation=cv.INTER_NEAREST)
        # Binarize image
        imgBinarized = (imgResized > 1) * 255
        imgBinarized[imgBinarized == 255] = 1

        # Save image
        np.save(templatePath + "num_" + str(i) + ".npy", imgBinarized)

    print("Done!")


main()
