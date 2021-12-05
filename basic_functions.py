import cv2 as cv

# convert to grayscale
img = cv.imread('./Resources/Photos/cat.jpg')
cv.imshow('Cat1', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

# blur an image
blurred = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
#cv.imshow('Blur', blurred)

# edge cascade
edge = cv.Canny(blurred, 125, 175)
#cv.imshow('Edges', edge)

# dialating
dialated = cv.dilate(edge, (7, 7), iterations=3)
#cv.imshow('Dialated', dialated)

# eroding
eroded = cv.erode(dialated, (7, 7), iterations=3)
#cv.imshow('Eroded', eroded)

# resize and crop
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)  # area for making smaller, INTER_LINEAR and INTER_CUBIC for upscaling
#cv.imshow('Resized', resized)

# cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)