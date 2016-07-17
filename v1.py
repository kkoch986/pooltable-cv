
import cv2
import numpy as np
import operator
from numpy import pi, sin, cos
import math

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
		self.hueThreshold = [8,23]
		self.medianKernelSize = 5
		self.maxHue = 0
		self.bottomRightIndex = 3
		self.bottomLeftIndex = 0
		self.rotationDirection = -1
		self.tableRectangle = False
		self.rotationTransform = False
		self.rotationCenter = None
		self.rotationAngle = 0
		self.perspectivePoints = False
	
	# captureSource
	def getCaptureSource(self):
		return self.captureSource

	# Capture
	def setCapture(self, cap):
		self.capture = cap
	def getCapture(self):
		return self.capture

	# medianKernelSize
	def setMedianKernelSize(self, val):
		self.medianKernelSize = val
	def getMedianKernelSize(self):
		return self.medianKernelSize

	# hueThreshold
	def getHueThreshold(self):
		return self.hueThreshold
	def setLowHueThreshold(self, val):
		self.hueThreshold[0] = val
	def setHighHueThreshold(self, val):
		self.hueThreshold[1] = val

	# maxHue
	def setMaxHue(self, val):
		self.maxHue = val
	def getMaxHue(self):
		return self.maxHue

	# bottomRightIndex
	def setBottomRightIndex(self, val):
		self.bottomRightIndex = val
	def getBottomRightIndex(self):
		return self.bottomRightIndex

	# bottomLeftIndex
	def setBottomLeftIndex(self, val):
		self.bottomLeftIndex = val
	def getBottomLeftIndex(self):
		return self.bottomLeftIndex

	# rotationDirection
	def setRotationDirection(self, val):
		if val == 0:
			self.rotationDirection = -1
		else:
			self.rotationDirection = 1
	def getRotationDirection(self):
		return self.rotationDirection
	def getRotationDirectionTrackbarValue(self):
		if self.rotationDirection == -1:
			return 0
		else:
			return 1

	# tableRectangle
	def setTableRectangle(self, rect):
		self.tableRectangle = rect
		self.setPerspectivePoints(rect)
	def getTableRectangle(self):
		return self.tableRectangle

	# rotationTransform
	def setRotationTransform(self, rect):
		self.rotationTransform = rect
	def getRotationTransform(self):
		return self.rotationTransform

	# rotationCenter
	def setRotationCenter(self, center):
		self.rotationCenter = center
	def getRotationCenter(self):
		return self.rotationCenter

	# rotationAngle
	def setRotationAngle(self, angle):
		self.rotationAngle = angle
	def getRotationAngle(self):
		return self.rotationAngle

	# perspectivePoints
	def getPerspectivePoints(self):
		return self.perspectivePoints
	def setPerspectivePoints(self, val):
		self.perspectivePoints = val
	def setPerpectiveX1(self, val):
		self.perspectivePoints[0][0] = val
	def setPerpectiveX2(self, val):
		self.perspectivePoints[1][0] = val
	def setPerpectiveX3(self, val):
		self.perspectivePoints[2][0] = val
	def setPerpectiveX4(self, val):
		self.perspectivePoints[3][0] = val
	def setPerpectiveY1(self, val):
		self.perspectivePoints[0][1] = val
	def setPerpectiveY2(self, val):
		self.perspectivePoints[1][1] = val
	def setPerpectiveY3(self, val):
		self.perspectivePoints[2][1] = val
	def setPerpectiveY4(self, val):
		self.perspectivePoints[3][1] = val

########################################
# The first step will be to show the video 
# feed and allow the user to adjust the 
# position of the camera
########################################
class CameraAlignmentStep:
	def __init__(self, settings):
		self.settings = settings

	def run(self):
		capture = create_capture(self.settings.getCaptureSource())
		self.settings.setCapture(capture)
		while True:
			ret, img = capture.read()
			
			(h,w,c) = img.shape
			cv2.putText(img, "Align the camera and press ENTER.", (0,h-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
			cv2.putText(img, "Try to get as much of the table in the frame as posssible.", (0,h-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

			cv2.imshow('Camera Adjustment', img)

			ch = 0xFF & cv2.waitKey(1)
			if ch == 13:
				break
		cv2.destroyAllWindows()

########################################
# Capture an image from the camera
# Then run the processing function to extract a default contour.
# once that contour is found, give the user trackbars to adjust the parameters
########################################
class TableDetectionStep:
	def __init__(self, settings):
		self.settings = settings
		ret, self.image = settings.getCapture().read()
		self.preprocessImage(self.image)

	# convert the image to HSV and do other preprocessing
	# steps since the image wont change
	def preprocessImage(self, img):
		############################################
		# Step 1
		# read in the image and convert it to HSV
		############################################
		self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		self.height, self.width, self.channels = img.shape
		self.total_area = self.height * self.width

		############################################
		# Step 2
		# build a histogram of the hue with a range of 0-255 and 255 buckets
		############################################
		bins = [255]
		range = [0,255]
		histogram = cv2.calcHist([self.hsv],[0],None,bins,range)
		cv2.normalize(histogram,histogram,0,255,cv2.NORM_MINMAX)
		hist = np.int32(np.around(histogram))
		max_hue, value = max(enumerate(hist), key=operator.itemgetter(1))
		self.settings.setMaxHue(max_hue);

	# Process the image based on the parameters in the settigns object (and modified by the trackbars)
	def processImage(self, img):
		imgCopy = img.copy()

		############################################
		# Step 3
		# create a mask of only the colors in the hue range
		############################################
		max_hue = self.settings.getMaxHue()
		hue_threshold = self.settings.getHueThreshold()
		mask = cv2.inRange(self.hsv, np.array([max_hue-hue_threshold[0], 0, 0]), np.array([max_hue+hue_threshold[1], 255, 255]))

		############################################
		# Step 4
		# median filter the mask
		############################################
		filtered_mask = cv2.medianBlur(mask, self.settings.getMedianKernelSize());

		############################################
		# Step 5
		# find the contour around the table
		############################################
		# start by finding all of the contours in the filtered mask
		ret, thresh = cv2.threshold(filtered_mask,127,127,0)
		im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

		# take the convex hull of all of the contours
		convex_hulls = [cv2.convexHull(cnt) for cnt in contours]

		# find the biggest hull by area and extract it from the list of hulls
		maxArea = max([cv2.contourArea(cnt) for cnt in convex_hulls])
		contour = [cnt for cnt in convex_hulls if cv2.contourArea(cnt) >= maxArea][0]

		# find the minimum bounding rectangle
		rect = cv2.minAreaRect(contour)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(imgCopy,[box],-1,(0,255,0),3)

		# create a mask out of the bounding rectable
		contour_mask = np.zeros_like(imgCopy)
		cv2.drawContours(contour_mask, [box], 0, (255,255,255), cv2.FILLED)

		# Bitwise-AND mask and original image
		res = np.zeros_like(imgCopy) # Extract out the object and place into output image
		res[contour_mask == 255] = imgCopy[contour_mask == 255]

		############################################
		# Step 6
		# rotate the image to 90 degrees along the bottom
		############################################
		# compute the angle needed to square the rectangle along the bottom
		bottomLeft = (box[self.settings.getBottomLeftIndex()][0], box[self.settings.getBottomLeftIndex()][1])
		bottomRight = (box[self.settings.getBottomRightIndex()][0], box[self.settings.getBottomRightIndex()][1])

		# find the equation for the bottom line
		if (1.0 * bottomLeft[0] - bottomRight[0]) == 0:
			m = 1
		else:
			m = (1.0 * bottomLeft[1] - bottomRight[1]) / (1.0 * bottomLeft[0] - bottomRight[0])

		if m != 0:
			b = (bottomLeft[1]) - (m * bottomLeft[0])

			thirdPoint = (int((self.height - b) / m), self.height)
			cv2.drawContours(res, [contour], 0, (255,255,255), 2)
			cv2.line(res, thirdPoint, bottomRight, (0,0,255), 3)
			cv2.line(res, bottomRight, (bottomRight[0], self.height), (0,0,255), 3)
			cv2.line(res, thirdPoint, (bottomRight[0], self.height), (0,0,255), 3)
			cv2.circle(res, bottomLeft, 20, (0,255,0))
			cv2.circle(res, bottomRight, 20, (0,255,0))

			hypLen = math.sqrt( math.pow( (thirdPoint[0]-bottomRight[0]), 2 ) + math.pow( (thirdPoint[1]-bottomRight[1]), 2 ) )
			oppLen = math.sqrt( 0 + math.pow( (bottomRight[1]-self.height), 2 ) ) 
			if(hypLen == 0):
				angle = 0
			else: 
				angle = math.sin(oppLen / hypLen)
			center = thirdPoint

			self.settings.setRotationCenter(center)
			self.settings.setRotationAngle(angle)

			# rotate the image
			mat = cv2.getRotationMatrix2D(center, self.settings.getRotationDirection() * math.degrees(angle), 1)
			rotated = cv2.warpAffine(res, mat, (self.width, self.height))
			rotated_mask = cv2.warpAffine(contour_mask, mat, (self.width, self.height))
		else:
			mat = cv2.getRotationMatrix2D((0,0), 0, 1)
			rotated = res
			rotated_mask = contour_mask

		############################################
		# Step 7
		# for calibration purposes, create a big image out of all of the smaller images
		############################################
		images = [
			self.hsv, 
			cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 
			cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2RGB), 
			contour_mask,
			imgCopy, 
			res, 
			rotated_mask,
			rotated
		]

		imageLabels = [
			"HSV",
			"Hue Mask",
			"Filtered Mask",
			"Contour Mask",
			"Hull",
			"Extracted",
			"Rotated Mask",
			"Rotated Capture"
		]

		return (stack_images(images, imageLabels, self.width, self.height, 2), box, mat)

	def run(self):
		while True:
			img, rectangle, rotationTransform = self.processImage(self.image);
			cv2.imshow('Table Detection', img)

			cv2.createTrackbar('Selected Hue', 'Table Detection', self.settings.getMaxHue(), 255, self.settings.setMaxHue)
			cv2.createTrackbar('Low Hue Threshold', 'Table Detection', self.settings.getHueThreshold()[0], 127, self.settings.setLowHueThreshold)
			cv2.createTrackbar('High Hue Threshold', 'Table Detection', self.settings.getHueThreshold()[1], 127, self.settings.setHighHueThreshold)
			cv2.createTrackbar('Bottom Right Index', 'Table Detection', self.settings.getBottomRightIndex(), 3, self.settings.setBottomRightIndex)
			cv2.createTrackbar('Bottom Left Index', 'Table Detection', self.settings.getBottomLeftIndex(), 3, self.settings.setBottomLeftIndex)
			cv2.createTrackbar('Rotation Direction', 'Table Detection', self.settings.getRotationDirectionTrackbarValue(), 1, self.settings.setRotationDirection)

			ch = 0xFF & cv2.waitKey(1)
			if ch == 13:
				break

		cv2.destroyAllWindows()

		self.settings.setTableRectangle(rectangle)
		self.settings.setRotationTransform(rotationTransform)

########################################
# Show the live video feed masked by the rectangle
# and rotated by the rotation matrix
# then allow the user to adjust the lines to set the
# perspective
########################################
class TablePerspectiveStep:
	def __init__(self, settings):
		self.settings = settings

	def processImage(self, img):
		# apply the mask
		box = self.settings.getPerspectivePoints()
		contour_mask = np.zeros_like(img)
		cv2.drawContours(contour_mask, [box], 0, (255,255,255), cv2.FILLED)

		# Bitwise-AND mask and original image
		res = np.zeros_like(img) # Extract out the object and place into output image
		res[contour_mask == 255] = img[contour_mask == 255]

		# draw the liens on the cropped image
		cv2.drawContours(res, [box], 0, (0,255,0), 3)

		# apply the transform
		height, width, channels = res.shape
		rotated = cv2.warpAffine(res, self.settings.getRotationTransform(), (width, height))

		# apply the perspective based on the lines
		inputPts = np.float32(box)
		outputPts = np.float32([ [0, height], [0,0], [width, 0], [width, height] ])

		M = cv2.getPerspectiveTransform(inputPts,outputPts)
		perspective = cv2.warpPerspective(res,M,(width,height))

		return stack_images([img, res, rotated, perspective], ["Source", "Masked", "Rotated", "Perspective"], width, height, 2)

	def run(self):
		capture = self.settings.getCapture()
		while True:
			ret, img = capture.read()
			cv2.imshow('Perspective Adjustment', self.processImage(img))

			h,w,c = img.shape
			p = self.settings.getPerspectivePoints()
			cv2.createTrackbar('X1', 'Perspective Adjustment', p[0][0], w, self.settings.setPerpectiveX1)
			cv2.createTrackbar('Y1', 'Perspective Adjustment', p[0][1], h, self.settings.setPerpectiveY1)
			cv2.createTrackbar('X2', 'Perspective Adjustment', p[1][0], w, self.settings.setPerpectiveX2)
			cv2.createTrackbar('Y2', 'Perspective Adjustment', p[1][1], w, self.settings.setPerpectiveY2)
			cv2.createTrackbar('X3', 'Perspective Adjustment', p[2][0], w, self.settings.setPerpectiveX3)
			cv2.createTrackbar('Y3', 'Perspective Adjustment', p[2][1], w, self.settings.setPerpectiveY3)
			cv2.createTrackbar('X4', 'Perspective Adjustment', p[3][0], w, self.settings.setPerpectiveX4)
			cv2.createTrackbar('Y4', 'Perspective Adjustment', p[3][1], w, self.settings.setPerpectiveY4)

			ch = 0xFF & cv2.waitKey(2)
			if ch == 13:
				break
		cv2.destroyAllWindows()

########################################
# Run the program
########################################
settings = CalibrationSettings()

# 1.
# Have the user position the camera
CameraAlignmentStep(settings).run()

# 2.
# try to detect the contours of the table
TableDetectionStep(settings).run()

# 3.
# Show the rotated rectangle filter applied to the live video feed
# give the user the ability to tighen the lines on the edges of the
# rectangle to give the perspective, then apply the perspecive transform based on that
TablePerspectiveStep(settings).run()