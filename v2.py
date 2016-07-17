
import cv2
import numpy as np
import operator
from numpy import pi, sin, cos
import math
import json
import sys

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def stack_images(images, imageLabels, width, height, rowCount = 2):
	targetWidth = (width / (len(images) / rowCount))
	aspectRatio = (1.0 * height / width)
	targetHeight = int(targetWidth * aspectRatio)
	
	perRow = len(images) / rowCount
	rows = []
	currentRow = []
	index = 0
	for i in images:
		cv2.putText(i, imageLabels[index], (0,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 4)
		index = index+1
		i2 = cv2.resize(i, (targetWidth, targetHeight))
		currentRow.append(i2);
		if(len(currentRow) >= perRow):
			rows.append(currentRow)
			currentRow = []
	
	if(len(currentRow)):
		rows.append(currentRow)

	cols = [np.hstack(row) for row in rows]
	return np.vstack(cols)

########################################
# a function to create a video capture 
########################################
def create_capture(source = 0, fallback = None):
    '''source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'
    '''
    source = str(source).strip()
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )

    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try: cap = Class(**params)
        except: pass
    else:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CV_CAP_PROP_FPS, video_fps)
        # if 'size' in params:
        #     w, h = map(int, params['size'].split('x'))
        #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        if fallback is not None:
            return create_capture(fallback, None)
    return cap

########################################
# A Class for holding all of the calibration settings
########################################
class CalibrationSettings:
	def __init__(self):
		self.captureSource = 0
		self.capture = None
		self.perspectivePoints = []
	
	def save(self):
		f = open('settings.json', 'w')
		f.write(json.dumps({
			'perspectivePoints': self.perspectivePoints.tolist()
		}))
		f.close()

	def load(self, path):
		f = open(path, 'r')
		data = json.loads(f.read())
		f.close()

		self.perspectivePoints = np.int0(data['perspectivePoints'])

	# captureSource
	def setCaptureSource(self, val):
		self.captureSource = val
	def getCaptureSource(self):
		return self.captureSource

	# Capture
	def setCapture(self, cap):
		self.capture = cap
	def getCapture(self):
		return self.capture

	# perspectivePoints
	def getPerspectivePoints(self):
		return self.perspectivePoints
	def setPerspectivePoints(self, val):
		self.perspectivePoints = val
	def setPersepectiveX(self, index, val):
		self.perspectivePoints[index][0] = val
	def setPersepectiveY(self, index, val):
		self.perspectivePoints[index][1] = val

##
# A class which has a read function which always returns a copy of the same image
##
class ImageCapture:
	def __init__(self, path):
		self.img = cv2.imread(path)

	def read(self):
		return (True, self.img.copy())

########################################
# Show the live video feed masked by the rectangle
# and rotated by the rotation matrix
# then allow the user to adjust the lines to set the
# perspective
########################################
class TablePerspectiveStep:
	def __init__(self, settings):
		self.settings = settings

		if is_number(self.settings.getCaptureSource()):
			capture = create_capture(self.settings.getCaptureSource())
		else:
			capture = ImageCapture(self.settings.getCaptureSource())
		self.settings.setCapture(capture)
		self.point = 0

	def processImage(self, img):
		height, width, channels = img.shape

		# apply the perspective based on the lines
		box = self.settings.getPerspectivePoints()

		if len(box) == 0:
			box = np.int0([[0,0], [0, height], [width, height], [width, 0]])
			self.settings.setPerspectivePoints(box)

		inputPts = np.float32(box)
		outputPts = np.float32([ [0,0], [0, height], [width, height], [width, 0] ])
		cv2.drawContours(img, [box], 0, (0,255,0), 3)

		M = cv2.getPerspectiveTransform(inputPts,outputPts)
		perspective = cv2.warpPerspective(img,M,(width,height))

		return stack_images([img, perspective], ["Source", "Perspective"], width, height, 1)

	def setPoint(self, val):
		self.point = val
		cv2.setTrackbarPos('X', 'Perspective Adjustment', self.settings.getPerspectivePoints()[self.point][0])
		cv2.setTrackbarPos('Y', 'Perspective Adjustment', self.settings.getPerspectivePoints()[self.point][1])

	def setPersepectiveX(self, val):
		self.settings.setPersepectiveX(self.point, val)

	def setPersepectiveY(self, val):
		self.settings.setPersepectiveY(self.point, val)

	def run(self):
		capture = self.settings.getCapture()
		while True:
			ret, img = capture.read()
			cv2.imshow('Perspective Adjustment', self.processImage(img))

			h,w,c = img.shape
			p = self.settings.getPerspectivePoints()

			index = self.point

			cv2.createTrackbar('Point', 'Perspective Adjustment', self.point, 3, self.setPoint)
			cv2.createTrackbar('X', 'Perspective Adjustment', p[self.point][0], w, self.setPersepectiveX)
			cv2.createTrackbar('Y', 'Perspective Adjustment', p[self.point][1], h, self.setPersepectiveY)

			ch = 0xFF & cv2.waitKey(2)

			if ch == 115:
				self.settings.save()

			if ch == 13:
				break
		cv2.destroyAllWindows()

# configure the settings
# load from argv[1] if its set
settings = CalibrationSettings()
if len(sys.argv) > 1:
	settings.load(sys.argv[1])
settings.setCaptureSource()
# settings.setCaptureSource("./source/dark1.jpg")

# allow the user to adjust the table perspective
TablePerspectiveStep(settings).run()
