import time
import cv2
import mediapipe as mp
import math
from screeninfo import get_monitors
from pynput.mouse import Button, Controller

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
  max_num_hands = 1,
  min_detection_confidence = 0.5
)

cap = cv2.VideoCapture(0)
mouse = Controller()
prevTime = 0
set = 0

def getScreenCoordsFromNormalized(image, normalized):
  image_height = image.shape[0]
  image_width = image.shape[1]

  # Get normalized coords inside the image
  imagePixelCoords = mp_drawing._normalized_to_pixel_coordinates(normalized.x, normalized.y, image_width, image_height)

  if type(imagePixelCoords) != tuple:
    return False

  # Convert the ratio from the coords inside the image
  # to fit with the screen
  screenPixelCoords = [imagePixelCoords[0] * screen_width / image_width, imagePixelCoords[1] * screen_height / image_height]

  return screenPixelCoords

for m in get_monitors():
  if m.is_primary:
    global screen_height, screen_width
    screen_width = m.width
    screen_height = m.height

while cap.isOpened():
  success, image = cap.read()

  if not success:
    print('Ignoring empty camera frame')
    continue

  image = cv2.flip(image, 1)
  results = hands.process(image)

  if results.multi_hand_landmarks:
    hand_landmarks = results.multi_hand_landmarks[0]
      
    mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Get normalized landmarks
    # CURSOR: hand part that is gonna lead where the cursor goes
    # TONE: first hand part that is used to trigger click
    # TTWO: second hand part that is used to trigger click
    CURSOR = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    TONE = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    TTWO = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Get screen coordinates IN from normalized landmarks
    CURSOR = getScreenCoordsFromNormalized(image, CURSOR)
    TONE = getScreenCoordsFromNormalized(image, TONE)
    TTWO = getScreenCoordsFromNormalized(image, TTWO)

    # If CURSOR or TONE or TTWO went outside camera
    if CURSOR != False and TONE != False and TTWO != False:
      # Add limits to cursor position to prevent it
      # going outside of screen
      if CURSOR[0] > screen_width:
        CURSOR[0] = screen_width
      elif CURSOR[0] < 0:
        CURSOR[0] = 0

      if CURSOR[1] > screen_height:
        CURSOR[1] = screen_width
      elif CURSOR[1] < 0:
        CURSOR[1] = 0
        
      mouse.position = CURSOR #(destX, destY)

      # Measure distances between TONE and TTWO
      # using pythagorean theorem
      distance = math.hypot(abs(TONE[0] - TTWO[0]), abs(TONE[1] - TTWO[1]))
      
      # Touch sensitivity of the click
      if distance < 80:
        mouse.click(Button.left)

      print(distance)

  currTime = time.time()
  fps = 1 / (currTime - prevTime)
  prevTime = currTime

  cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
  cv2.imshow('Hand tracker', image)

  if cv2.waitKey(5) & 0xFF == 27:
    break

cap.release()

