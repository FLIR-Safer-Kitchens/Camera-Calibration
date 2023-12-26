import numpy as np
import cv2


# Temp. Conversion Helpers
f2c = lambda f: 5/9*(f-32)
c2f = lambda c: 9/5*c + 32


# Radiometry Conversion Helpers (temp units = C)
temp2raw = lambda t: round(100*t)+27315
raw2temp = lambda r: (r-27315)/100


# Takes a .tiff file and returns sequential frames similar to cv2.VideoCapture 
class Raw16Video:
	# Load .tiff file. Lepton doesn't chunk files
	def __init__(self, filename):
		ret, self.frames = cv2.imreadmulti(
			filename=filename, 
			flags=cv2.IMREAD_UNCHANGED
		)
		assert ret, "Failed to read .tiff file"
		self.frames = iter(self.frames)

	# Try to return the next frame
	# Otherwise return False
	def read(self):
		try: return True, next(self.frames)
		except StopIteration: return False, None


# Saves frames to video file
class SaveVideo:
	# Create VideoWriter Object
	def __init__(self, filename, color=False):
		self.__vid = cv2.VideoWriter(
			filename=filename,
			apiPreference=cv2.CAP_FFMPEG,
			fourcc=cv2.VideoWriter_fourcc(*'XVID'),
			fps=8.0,
			frameSize=(160, 120),
			params=[
				cv2.VIDEOWRITER_PROP_IS_COLOR, int(color)
			],
		)
		
		assert self.__vid.isOpened()

	# Destructor
	def __del__(self):
		self.close()

	# Write a new frame to video
	def write(self, frame):
		assert self.__vid.isOpened()
		self.__vid.write(frame)

	# Close the video file
	def close(self):
		if self.__vid.isOpened():
			self.__vid.release()


# Limit pixel values to a given 16-bit range
# Convert 16-bit values to 8-bit
def clip_norm(img, min_val=None, max_val=None):
	if min_val == None: min_val = int(min(img.flatten()))
	if max_val == None: max_val = int(max(img.flatten()))

	# Clip image
	img = np.clip(img.flatten(), min_val, max_val).reshape(img.shape)

	# Map 16-bit to 8-bit
	# Surely there's a function for this
	img = cv2.multiply(cv2.subtract(img, min_val), 255/(max_val-min_val+0.001))
	return img.astype('uint8')


# Histogram Equalization
# clipped = True --> ignore first and last bins 
def hist_equalize(img, clipped=False):
	# Get histogram and CDF
	hist = np.histogram(img.flatten(), 256, [0, 255])[0]
	if clipped: hist[0] = hist[-1] = 0
	cdf = hist.cumsum()

	# Create LUT based on linear CDF
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')

	return cdf[img]
