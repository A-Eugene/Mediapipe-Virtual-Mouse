import time
import cv2
import mediapipe as mp
import math
import json
from screeninfo import get_monitors
from pynput.mouse import Button, Controller
from threading import Thread

config = json.loads(open('config.json', 'r').read())

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
  max_num_hands = 1,
  min_detection_confidence = config['minimum_confidence']
)

cap = cv2.VideoCapture(0)
mouse = Controller()
prevTime = 0
setted = 0
lastTimeWasClick = False
lastCursorPosition = (0, 0)

def getScreenCoordsFromNormalized(image, normalized):
  # Get normalized coords inside the image
  imagePixelCoords = mp_drawing._normalized_to_pixel_coordinates(normalized.x, normalized.y, image_width, image_height)

  if type(imagePixelCoords) != tuple:
    return False

  pxX = imagePixelCoords[0]
  pxY = imagePixelCoords[1] 

  # Add limits to cursor position to prevent it
  # going outside of screen
  
  # STEPS:
  # 1. Get the relative tracked position to the image from the
  # normalized landmark
  # 2. If the tracked position is outside the controlarea,
  # then we can assure that the equivalent of that position
  # in the screen would be 0 if it is below the controlarea's startpoint, or
  # %length of the screen dimension% if it is over controlarea's endpoint 
  # 3. Else if the tracked position is inside the controlarea,
  # we can get the tracked position relative to the controlarea by calculating:
  #
  # %tracked position relative to the image% - %controlarea's startpoint%
  #
  # 4. We shall convert the tracked position relative to the controlarea to
  # the equivalent of that position in the screen by calculating:
  #
  # %tracked position relative to the controlarea% * %screen's dimension% / %controlarea's dimension%
  #
  # just a basic ratio conversion
  # (This is done to make sure the ability to control 
  # the cursor reaches the edges of the screen)
  #
  # DETAILS:
  # - (pxX - boxStartX): position of the tracked component relative to controlarea's left edge
  # - (pxY - boxStartY): position of the tracked component relative to controlarea's top edge
  # - (image_width * config['controlarea_to_image_scale']): controlarea's width
  # - (image_height * config['controlarea_to_image_scale']): controlarea's height
  #
  if pxX < boxStartX:
    x = 0
  elif pxX > boxEndX:
    x = screen_width
  else:
    x = (pxX - boxStartX) / (image_width * config['controlarea_to_image_scale']) * screen_width 

  if pxY < boxStartY:
    y = 0
  elif pxY > boxEndY:
    y = screen_height
  else:
    y = (pxY - boxStartY) / (image_height * config['controlarea_to_image_scale']) * screen_height 

  return [x, y]

for m in get_monitors():
  if m.is_primary:
    global screen_height, screen_width
    screen_width = m.width
    screen_height = m.height
    break

while cap.isOpened():
  success, image = cap.read()

  # Excute only once
  if setted == 0:
    global boxStartX, boxStartY, boxEndX, boxEndY, image_height, image_width, controlarea_width, controlarea_height
    
    image_height = image.shape[0]
    image_width = image.shape[1]
    controlarea_width = image_width * config['controlarea_to_image_scale']
    controlarea_height = image_width * config['controlarea_to_image_scale']
    boxStartX = image_width * (1 - config['controlarea_to_image_scale']) / 2
    boxStartY = image_height * (1 - config['controlarea_to_image_scale']) / 2
    boxEndX = boxStartX + (config['controlarea_to_image_scale'] * image_width)
    boxEndY = boxStartY + (config['controlarea_to_image_scale'] * image_height)

  if not success:
    print('Ignoring empty camera frame')
    continue

  image = cv2.flip(image, 1)
  results = hands.process(image)

  currTime = time.time()
  fps = 1 / (currTime - prevTime)
  prevTime = currTime

  if results.multi_hand_landmarks:
    hand_landmarks = results.multi_hand_landmarks[0]
      
    mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Get normalized landmarks
    # CURSOR: hand part that is gonna lead where the cursor goes
    # TONE: first hand part that is used to trigger click
    # TTWO: second hand part that is used to trigger click
    CURSOR = hand_landmarks.landmark[mp_hands.HandLandmark[config['CURSOR']]]
    TONE = hand_landmarks.landmark[mp_hands.HandLandmark[config['TONE']]]
    TTWO = hand_landmarks.landmark[mp_hands.HandLandmark[config['TTWO']]]

    # Get screen coordinates IN from normalized landmarks
    CURSOR = getScreenCoordsFromNormalized(image, CURSOR)
    TONE = getScreenCoordsFromNormalized(image, TONE)
    TTWO = getScreenCoordsFromNormalized(image, TTWO)

    # If CURSOR or TONE or TTWO went outside camera
    if CURSOR and TONE and TTWO != False: 
      mouse.position = CURSOR

      # Measure distances between TONE and TTWO
      # using pythagorean theorem
      Tdistance = math.hypot(abs(TONE[0] - TTWO[0]), abs(TONE[1] - TTWO[1]))
      
      # Determine whether a certain distance
      # Between handparts is enough to click
      if Tdistance < config['minimum_trigger_distance']:
        if lastTimeWasClick == False:
          mouse.click(Button.left)
          lastTimeWasClick = True
      # Ignores lastTimeWasClick's state
      else:
        lastTimeWasClick = False

      print(Tdistance)

  cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
  cv2.rectangle(image, (int(boxStartX), int(boxStartY)), (int(boxEndX), int(boxEndY)), (230, 230, 48), 2)
  cv2.imshow('Hand tracker', image)

  if cv2.waitKey(5) & 0xFF == 27:
    break

cap.release()

