import cv2 as cv

def rescaleFrame(frame, scale=.75):
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)
  dimensions = (width, height)
  return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('./Resources/Photos/cat_large.jpg')
cv.imshow('Image', rescaleFrame(img, .2))

capture = cv.VideoCapture('./Resources/Videos/dog.mp4')

while True:
  isTrue, frame = capture.read() # returns if frame is successfully read in or not + the frame
  frame_resized = rescaleFrame(frame, .25)
  cv.imshow('Video', frame_resized)
  
  if cv.waitKey(20) & 0xFF == ord('d'):  # 0xFF == ord('d') -> check if letter d is pressed
    break

capture.release()
cv.destroyAllWindows()
