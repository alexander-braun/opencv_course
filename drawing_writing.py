import cv2 as cv
import numpy as np

# zeros((width, height, color-channels), data-types)
blank = np.zeros((500, 500, 3), dtype='uint8') # uint8 == data type of an image
# cv.imshow('Blank', blank)

# Paint the image a certain colour
blank[:] = 50,155,100
#cv.imshow('Green', blank)

# Draw a rectangle ( image, point1, point2, color, thickness)
cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), 2)
#cv.imshow('Rectangle', blank)

# Draw a circle (filled)
cv.circle(blank, (250, 250), 150, (100, 200, 230), cv.FILLED)
#cv.imshow('Circle', blank)

# Draw a line
cv.line(blank, (200, 200), (300, 300), (255, 255, 255), 40)
#cv.imshow('Line', blank)

# Text
cv.putText(blank, "Hello there!", (255, 255), cv.FONT_HERSHEY_TRIPLEX, 1, (10, 10, 10), 4)
cv.imshow('Text', blank)


cv.waitKey(0)