# source: https://www.youtube.com/watch?v=eLTLtUVuuy4
import cv2
import numpy as np


def canny(image):
    # Convert the image copy to grayscale
    gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)

    # Reduce image noise using Gaussian blur
    blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # Apply canny method to find edges in the image
    # Calculating the gradient in the image
    return cv2.Canny(blur, threshold1=50, threshold2=150)


def region_of_interest(image):
    height = image.shape[0]
    length = image.shape[1]
    # Area of interest in the image
    polygons = np.array([
        [(0, height), (length, height), (int(length/2), int(height/2))]
    ])

    # Create a copy of the image with black pixels
    mask = np.zeros_like(image)

    # Fill the mask with our region of interest in white pixels
    mask = cv2.fillPoly(mask, pts=polygons, color=255)

    # Overlay original image with the mask
    masked_image = cv2.bitwise_and(src1=image, src2=mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)
                # Check whether the coordinates are in inside the image, and draw
                if (x1 >= 0 and x1 <= image.shape[1]) and (x2 >= 0 and x2 <= image.shape[1]) \
                            and (y1 >= 0 and y1 <= image.shape[0]) and (y2 >= 0 and y2 <= image.shape[0]):
                    cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=10)

    return line_image


def make_coordinates(image, line_parameters):
    # Create coordinates for the right and left lines on the road
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/4))

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    # Average out the lines slope calculate a single line on left and right side of the road
    left_fit = []
    right_fit = []
    left_line = None
    right_line = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit(x=(x1, x2), y=(y1, y2), deg=1)
            slope = parameters[0]
            intercept = parameters[1]
            # Right side line has positive slope and vise versa
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        # Check for whether a line exists
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(image, left_fit_average)

        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(image, right_fit_average)

        return np.array([left_line, right_line])

    else:
        return None

"""
# Import image using openCV
#raw_image = cv2.imread('test_image.jpg')
raw_image = cv2.imread('maxresdefault.jpg')
lane_image = np.copy(raw_image)

canny_image = canny(lane_image)

cropped_image = region_of_interest(canny_image)

# Identify lines in our image using Hough transform (y = mx+b => (m,b))
lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=100, \
                        lines=np.array([]), minLineLength=40, maxLineGap=5)

# Optimization
averaged_line = average_slope_intercept(lane_image, lines)

line_image = display_lines(raw_image, averaged_line)

combo_image = cv2.addWeighted(src1=lane_image, alpha=1, src2=line_image, beta=0.8, gamma=1)

# Show image
#plt.imshow(canny)
#plt.show()
cv2.imshow('result', combo_image)
cv2.waitKey(0)
"""

cap = cv2.VideoCapture("road_video.mp4")

while cap.isOpened():
    _, frame = cap.read()
    resized_frame = frame
    #resized_frame = frame[:, 50:]

    print(resized_frame.shape)

    canny_image = canny(resized_frame)
    cropped_image = region_of_interest(canny_image)

    # Identify lines in our image using Hough transform (y = mx+b => (m,b))
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=100, \
                            lines=np.array([]), minLineLength=40, maxLineGap=5)

    # Optimization
    averaged_line = average_slope_intercept(resized_frame, lines)

    line_image = display_lines(resized_frame, averaged_line)

    # Display lines on our original frame
    combo_image = cv2.addWeighted(src1=resized_frame, alpha=1, src2=line_image, beta=1, gamma=1)

    cv2.imshow('result', combo_image)
    if cv2.waitKey(10) == ord('q'):
        break

# Close video file
cap.release()
cv2.destroyAllWindows()

