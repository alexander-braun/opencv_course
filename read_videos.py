import cv2 as cv

# Reading videos
capture = cv.VideoCapture('./Resources/Videos/dog.mp4')


while True:
  isTrue, frame = capture.read() # returns if frame is successfully read in or not + the frame
  cv.imshow('Video', frame)
  
  if cv.waitKey(20) & 0xFF == ord('d'):  # 0xFF == ord('d') -> check if letter d is pressed
    break

capture.release()
cv.destroyAllWindows()