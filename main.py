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
    copyImg = drawImage(imgCBS, 255)

    # showImg(copyImg)


# Bresenham's Cicle
def drawImage(img, color):
    xc, yc = np.shape(img)
    xc = int(xc / 2)
    yc = int(yc / 2)
    r = int(0.8 * np.amin([xc, yc]))

    d = 3 - (2 * r)
    x = 0
    y = r

    # Points could be our line
    x_points = np.array([])
    y_points = np.array([])

    while(True):
        x_points, y_points = controlOnCircle(img, xc, yc, x, y, x_points, y_points)
        if (x <= y):
            x += 1
            if (d < 0):
                d = d + (4 * x) + 6
            else:
                d = d + 4 * (x - y) + 10 
                y -= 1
        else:
            break
    print(x_points)  # That's x points of the lines or any points
    print(y_points)  # That's y points of the lines or any points
    return img


def controlOnCircle(img, xc, yc, x, y, x_points, y_points):
    if (img[xc + x][yc + y] == 255):            # 1 of octans
        x_points = np.append(x_points, xc + x)
        y_points = np.append(y_points, yc + y)

    elif (img[xc - x][yc + y] == 255):          # 2 of octans
        x_points = np.append(x_points, xc - x)
        y_points = np.append(y_points, yc + y)

    elif (img[xc + x][yc - y] == 255):          # 3 of octans
        x_points = np.append(x_points, xc + x)
        y_points = np.append(y_points, yc - y)

    elif (img[xc - x][yc - y] == 255):          # 4 of octans
        x_points = np.append(x_points, xc - x)
        y_points = np.append(y_points, yc - y)

    elif (img[xc + y][yc + x] == 255):          # 5 of octans
        x_points = np.append(x_points, xc + y)
        y_points = np.append(y_points, yc + x)

    elif (img[xc - y][yc + x] == 255):          # 6 of octans
        x_points = np.append(x_points, xc - y)
        y_points = np.append(y_points, yc + x)   

    elif (img[xc + y][yc - x] == 255):          # 7 of octans
        x_points = np.append(x_points, xc + y)
        y_points = np.append(y_points, yc - x)

    elif (img[xc - y][yc - x] == 255):          # 8 of octans
        x_points = np.append(x_points, xc - y)
        y_points = np.append(y_points, yc - x)

    return [x_points, y_points]

# def drawCircle(img,xc, yc, x, y, color):
#     img[xc + x][yc + y] = color  # 1 of octans
#     img[xc - x][yc + y] = color  # 2 of octans
#     img[xc + x][yc - y] = color  # 3 of octans
#     img[xc - x][yc - y] = color  # 4 of octans
#     img[xc + y][yc + x] = color  # 5 of octans
#     img[xc - y][yc + x] = color  # 6 of octans
#     img[xc + y][yc - x] = color  # 7 of octans
#     img[xc - y][yc - x] = color  # 8 of octans
#     return img




def showImg(img):
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def Canny(img):
    edges = cv.Canny(img, 50, 150)
    return edges


if __name__ == "__main__":
    main()
