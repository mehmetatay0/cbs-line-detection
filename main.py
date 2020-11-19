from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd


def main():
    img = cv.imread('img/power-line.jpg')
    
    # Canny image filter applied
    filtered = Canny(img)

    # Step of Circle Based Search
    imgCBS = filtered.copy()
    copyImg = drawImage(imgCBS, 150, 150, 100, 255)
    
    showImg(copyImg)


# Bresenham's Cicle
def drawImage(img, xc, yc, r, color):
    d = 3 - (2 * r)
    x = 0
    y = r
    while(True):
        img = drawCircle(img, xc, yc, x, y, color)
        if (x <= y):
            x += 1
            if (d < 0):
                d = d + (4 * x) + 6
            else:
                d = d + 4 * (x - y) + 10 
                y -= 1
        else:
            break
    return img


def drawCircle(img,xc, yc, x, y, color):
    img[xc + x][yc + y] = color  # 1 of octans
    img[xc - x][yc + y] = color  # 2 of octans
    img[xc + x][yc - y] = color  # 3 of octans
    img[xc - x][yc - y] = color  # 4 of octans
    img[xc + y][yc + x] = color  # 5 of octans
    img[xc - y][yc + x] = color  # 6 of octans
    img[xc + y][yc - x] = color  # 7 of octans
    img[xc - y][yc - x] = color  # 8 of octans
    return img


def showImg(img):
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def Canny(img):
    edges = cv.Canny(img, 50, 150)
    return edges


if __name__ == "__main__":
    main()
