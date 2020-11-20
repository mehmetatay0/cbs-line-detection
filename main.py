from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd


def main():
    img = cv.imread('img/power-line.jpg')
    img_copy = img.copy()
    # Canny image filter applied
    filtered = Canny(img)

    # Step of Circle Based Search
    imgCBS = filtered.copy()
    x_points, y_points = drawImage(imgCBS, 255)

    for i in range(len(x_points)):
        resultImg = cv.circle(img, (y_points[i], x_points[i]), 5, (255, 0, 0), thickness=2)
    xc, yc = np.shape(imgCBS)
    xc = int(xc / 2)
    yc = int(yc / 2)
    r = int(0.8 * np.amin([xc, yc]))
    resultImg = cv.circle(resultImg, (yc, xc), r, (255,255,0), thickness=2)


    line_xy0_points, line_xy1_points = findLine(imgCBS, x_points, y_points)
    resultImg = cv.line(resultImg, (int(line_xy0_points[1]),int(line_xy0_points[0])), (int(line_xy1_points[1]), int(line_xy1_points[0])), (0, 255, 0), thickness=5)
    
    cv.imwrite('results/result_of_power-line.jpg', resultImg)
    cv.imshow("img", img_copy)
    cv.imshow("img2", resultImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Bresenham's Cicle Algorithm
def drawImage(img, color):
    xc, yc = np.shape(img)
    xc = int(xc / 2)
    yc = int(yc / 2)
    r = int(0.8 * np.amin([xc, yc]))

    d = 3 - (2 * r)
    x = 0
    y = r

    # Points could be our line
    x_points = np.array([]) # That's x points of the lines or any points
    y_points = np.array([]) # That's y points of the lines or any points

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
    return [x_points.astype(int), y_points.astype(int)]


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


# DDA Line Algorithm
def drawLine(img, x0, y0, x1, y1, line_xy0_points, line_xy1_points):
    match = 0
    X0 = x0
    Y0 = y0
    X1 = x1
    Y1 = y1
    dx = x1 - x0
    dy = y1 - y0

    length = abs(dx) if abs(dx) > abs(dy) else abs(dy)
    xInc = dx/float(length)
    yInc = dy/float(length)

    for i in range(length):
        if (img[ int(x0) ][ int(y0) ] == 255):
            match += 1
        x0 += xInc
        y0 += yInc
    if(match > 100):
        line_xy0_points = np.append(line_xy0_points, [X0, Y0], axis=0)
        line_xy1_points = np.append(line_xy1_points, [X1, Y1], axis=0)
    return [line_xy0_points, line_xy1_points]


def findLine(img, x_points, y_points):
    line_xy0_points = np.array([])
    line_xy1_points = np.array([])
    for i in range(len(x_points)):
        for j in range(len(x_points)):
            if (i != j):
                line_xy0_points, line_xy1_points = drawLine(img, x_points[i], y_points[i], x_points[j], y_points[j], line_xy0_points, line_xy1_points)
    return [line_xy0_points, line_xy1_points]


def Canny(img):
    edges = cv.Canny(img, 50, 150)
    return edges


if __name__ == "__main__":
    main()
