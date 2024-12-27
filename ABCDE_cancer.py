import cv2
import numpy as np
import imutils
import math
import copy


img = cv2.imread('ISIC_0272509M6 - Copy.jpg')

copy4=img.copy()

def rescaleFrame(frame, scale=0.1):
    #for images and vids
    width = int(frame.shape[1] * scale)
    height = int( frame.shape[0]*scale)

    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation= cv2.INTER_AREA)

resized_image = rescaleFrame(img)

cv2.imshow('Resized Image', resized_image)

gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

assert img is not None, "file could not be read, check with os.path.exists()"


# Gamma correction function

def gamma_correction(image, gamma=3):
    # Apply gamma correction
    corrected_image = np.uint8(((image / 255.0) ** gamma) * 255)
    return corrected_image

# Apply gamma correction (adjust gamma value as needed)
gamma_value = 3 # Adjust gamma value here
corrected_img = gamma_correction(gray, gamma_value)

ret1,th1 = cv2.threshold(corrected_img,10,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(corrected_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(corrected_img,(251,251),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms


contours, hierarchy = cv2.findContours(image=th3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# create an empty mask
mask = np.zeros(corrected_img.shape[:2], dtype=np.uint8)

# loop through the contours
for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it
    if hierarchy[0][i][2] == -1:
        # if the size of the contour is greater than a threshold
        if cv2.contourArea(cnt) > 100000:
            cv2.drawContours(mask, [cnt], 0, (255), -1)
        # display result

"""cv2.imshow("Img", corrected_img)"""

"""contours, hierarchy = cv.findContours(np.array(th3, dtype=np.int32), cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)

#Draw the contours on the result image
cv.drawContours(img, contours, -1, (0, 255, 0), 2)"""

"""resized_image4 = rescaleFrame4(mask)"""
resized_image1 = rescaleFrame(mask)

"""cv2.imshow('resized', resized_image1)"""

copy2=resized_image1.copy()


# Find contours
contours, hierarchy = cv2.findContours(resized_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(resized_image, contours, -1, (255, 255, 0), 2)

copy1=resized_image.copy()


# Calculate asymmetry for each contour
for contour in contours:
    # Compute the area of the contour
    area = cv2.contourArea(contour)

    # Find the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        # Print the centroid coordinates
        """print("Centroid Coordinates:", centroid_x, centroid_y)"""

        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)  # Green dot

        # Define the boundaries of the ROI
        x, y, w, h = cv2.boundingRect(contour)
        roi = corrected_img[y:y+h, x:x+w]

        # Compute the centroid of the ROI
        roi_centroid_x = centroid_x - x
        roi_centroid_y = centroid_y - y

        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (x + roi_centroid_x, y+1), 5, (0, 255, 0), -1)  # Green dot
        """print("Coordinates Of Top Point:", x + roi_centroid_x, y+2)"""

        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (x + roi_centroid_x, y + h-3), 5, (0, 255, 0), -1)  # Green dot
        """print("Coordinates Of Bottom Point :", x + roi_centroid_x, y + h-2)"""

        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (x+1, y + roi_centroid_y), 5, (0, 255, 0), -1)  # Green dot
        """print("Coordinates Of Left Point :", x, y + roi_centroid_y)"""

        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (x -2 + w, y + roi_centroid_y), 5, (0, 255, 0), -1)  # Green dot
        """print("Coordinates Of Right Point :", x -1+ w, y + roi_centroid_y)"""

        Radius_Top = centroid_y - (y + 1)
        """print("Radius Top: ", Radius_Top)"""

        Radius_Bottom = (y + h - 3) - centroid_y
        """print("Radius Bottom: ", Radius_Bottom)"""

        Radius_Left = centroid_x - (x+1)
        """print("Radius Left: ", Radius_Left)"""

        Radius_Right = (x - 5 + w) - centroid_x
        """print("Radius Right: ", Radius_Right)"""

        Difference1 = np.abs(Radius_Top - Radius_Bottom)
        """print("Difference between top and bottom radius: ", Difference1)"""

        Difference2 = np.abs(Radius_Left - Radius_Right)
        """print("Difference between right and left radius: ", Difference2)"""

        Total = Difference1 + Difference2
        """print("Total difference: ", Total)"""

# Display the image with contours and centroid dots
cv2.imshow('Borders and points drawn: ', resized_image)


"""print("Number of contours in image:",len(contours))"""
cnt=contours[0]

area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)
perimeter = round(perimeter, 4)
"""print('Area:', area)
print('Perimeter:', perimeter)"""
img1 = cv2.drawContours(resized_image, contours, -1, (255, 255, 0), 2)
x1, y1 = cnt[0,0]
cv2.putText(img1, f'Area: {area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255), 2)
cv2.putText(img1, f'Perimeter: {perimeter}', (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255), 2)


circularity = 4 * np.pi * area / (perimeter * perimeter)
"""print('Circularity:', circularity)"""

asymmetry= Total/circularity


# Initialize the ASYMMETRY variable
ASYMMETRY = None

# Print the label for asymmetry index
print("New Asymmetry index: ", end="")

# Check asymmetry value and assign the asymmetry index
if asymmetry <=1:
    ASYMMETRY = 1
elif 1 < asymmetry < 3:
    ASYMMETRY = 2
elif 3 < asymmetry < 6:
    ASYMMETRY = 3
elif 6 < asymmetry < 9:
    ASYMMETRY = 4
elif 9 < asymmetry < 12:
    ASYMMETRY = 5
elif 12 < asymmetry < 15:
    ASYMMETRY = 6
elif 15 < asymmetry < 18:
    ASYMMETRY = 7
elif 18 < asymmetry < 21:
    ASYMMETRY = 8
elif 21 < asymmetry < 24:
    ASYMMETRY = 9
else:
    ASYMMETRY = 10

# Print the assigned ASYMMETRY value
print(ASYMMETRY)


"""cv2.imshow('resized', resized_image)"""


cnt=contours[0]

area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)
perimeter = round(perimeter, 4)
"""print('Area:', area)
print('Perimeter:', perimeter)"""
img1 = cv2.drawContours(copy1, contours, -1, (255, 255, 0), 2)
x1, y1 = cnt[0,0]
"""cv2.putText(img1, f'Area: {area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255), 2)
cv2.putText(img1, f'Perimeter: {perimeter}', (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255), 2)
"""
x= pow(perimeter, 2)
compactness= x/(4*3.141592654*area)  # measuring compactness. perimeter^2/4pi*area
"""print('Compactness:', compactness)"""

"""circularity = 4 * np.pi * area / (perimeter * perimeter)
print('Circularity:', circularity)"""


# Initialize the BORDER variable
BORDER = None

# Check compactness value and assign Border Irregularity Index
if compactness == 1:
    BORDER = 1
elif 1 < compactness < 1.05:
    BORDER = 2
elif 1.05 <= compactness < 1.1:
    BORDER = 3
elif 1.1 <= compactness < 1.15:
    BORDER = 4
elif 1.15 <= compactness < 1.2:
    BORDER = 5
elif 1.2 <= compactness < 1.25:
    BORDER = 6
elif 1.25 <= compactness < 1.3:
    BORDER = 7
elif 1.3 <= compactness < 1.35:
    BORDER = 8
elif 1.35 <= compactness < 1.4:
    BORDER = 9
elif compactness >= 1.4:
    BORDER = 10

# Print the assigned BORDER value
print("New border Irregularity Index:", BORDER)


cv2.imshow('Borders drawn', copy1)


def count_colors(image):
    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image to a list of RGB tuples
    flattened_image = image_rgb.reshape(-1, 3)

    # Convert the RGB tuples to a set to get unique colors
    unique_colors = set(tuple(color) for color in flattened_image)

    # Count the number of unique colors
    num_colors = len(unique_colors)

    return num_colors


def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


# Function to check if a color is present in the image
def check_color_presence(image, color_lower, color_upper):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for the color
    lower = np.array(color_lower, dtype=np.uint8)
    upper = np.array(color_upper, dtype=np.uint8)
    # Threshold the HSV image to get only specified color region
    mask = cv2.inRange(hsv_image, lower, upper)
    # Count the number of non-zero pixels in the mask
    presence = cv2.countNonZero(mask)
    return 1 if presence > 0 else 0



path = 'finalISIC_0272509M6 - Copy.jpg'
# Read image
image55 = cv2.imread(path, cv2.IMREAD_COLOR)
# Image cropping
image = imutils.resize(image55, width=500, height=750)


# compute the colorfulness metric for the image
C = image_colorfulness(image)

# Count the number of colors in the image
num_colors = count_colors(image)

# Check presence of specific colors
light_brown = check_color_presence(image, [10, 50, 100], [30, 150, 255])
dark_brown = check_color_presence(image, [0, 20, 50], [20, 100, 150])
red = check_color_presence(image, [160, 100, 20], [179, 255, 255])
blue = check_color_presence(image, [101, 50, 38], [110, 255, 255])
white = check_color_presence(image, [0, 0,200], [180, 30, 255])

# Display colorfulness score and number of colors on the image
cv2.putText(image, "Colorfulness: {:.2f}".format(C), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(image, "Num Colors: {}".format(num_colors), (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Display presence of specific colors
cv2.putText(image, "Light Brown: {}".format(light_brown), (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(image, "Dark Brown: {}".format(dark_brown), (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(image, "Red: {}".format(red), (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(image, "Blue: {}".format(blue), (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(image, "White: {}".format(white), (10, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



img25 = cv2.imread('ISIC_0272509M6 - Copy.jpg')



def rescaleFrame(frame, scale=0.1):
    #for images and vids
    width = int(frame.shape[1] * scale)
    height = int( frame.shape[0]*scale)

    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation= cv2.INTER_AREA)

resized_image = rescaleFrame(img25)


"""cv2.imwrite('Photos/resizeddd.jpg', resized_image)"""

"""cv2.imshow('orig', resized_image)"""

gray= cv2.cvtColor(img25, cv2.COLOR_BGR2GRAY)

assert img is not None, "file could not be read, check with os.path.exists()"


# Gamma correction function

def gamma_correction(image, gamma=3):
    # Apply gamma correction
    corrected_image = np.uint8(((image / 255.0) ** gamma) * 255)
    return corrected_image

# Apply gamma correction (adjust gamma value as needed)
gamma_value = 3 # Adjust gamma value here
corrected_img = gamma_correction(gray, gamma_value)

ret1,th1 = cv2.threshold(corrected_img,10,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(corrected_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(corrected_img,(251,251),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms



contours, hierarchy = cv2.findContours(image=th3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# create an empty mask
mask1 = np.zeros(corrected_img.shape[:2], dtype=np.uint8)

# loop through the contours
for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it
    if hierarchy[0][i][2] == -1:
        # if the size of the contour is greater than a threshold
        if cv2.contourArea(cnt) > 100000:
            cv2.drawContours(mask1, [cnt], 0, (255), -1)
        # display result

"""cv2.imshow("Img", corrected_img)"""


"""contours, hierarchy = cv.findContours(np.array(th3, dtype=np.int32), cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)

#Draw the contours on the result image
cv.drawContours(img, contours, -1, (0, 255, 0), 2)"""

"""resized_image4=rescaleFrame4(mask1)"""
resized_image1 = rescaleFrame(mask1)

"""cv2.imshow('resized', resized_image1)"""
"""cv2.imwrite('Photos/whitefinalISIC_0341262.jpg', resized_image1)"""


replaced_image = cv2.bitwise_and(resized_image,resized_image,mask = resized_image1)
"""cv2.imshow('final', replaced_image)
cv2.imwrite('Photos/finalISIC_0341262.jpg', replaced_image)"""



hsv_img = cv2.cvtColor(replaced_image, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsv_image.jpg', hsv_img)


# black
low_BL = np.array([0, 0, 0])
upper_BL = np.array([350, 55, 100])
mask_BL = cv2.inRange(hsv_img, low_BL, upper_BL)

"""cv2.imshow('Black', mask_BL)
"""
# Here i am counting the number of black pixels first for the segmented image with a black background amd the entire mole in white
num_black_pixels_inoriginal = np.sum(resized_image1 == 0)
# Here i am counting the number of black pixels for the black background but now with the moles actual colours
num_black_pixels_innew = np.sum(mask_BL == 255)

"""print("Black pixel number without mole:  ", num_black_pixels_inoriginal)
print("Black pixel number with mole:  ", num_black_pixels_innew)
"""
# See if the pixel count has increased
if num_black_pixels_innew > num_black_pixels_inoriginal:
    x=1
else:
    x=0

cv2.putText(image, "Black: {}".format(x), (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# display the image
cv2.imshow("Mask around the lesion", image)

Total= (light_brown*1000)+(white*1000)+(red*1000)+(x*1000)+(dark_brown*1000)+(blue*1000)+C+num_colors


"""print("Total: ", Total)"""


# Initialize the COLOR variable
COLOR = None

# Check the value of Total and assign the Color Variegation Index
if Total < 5000:
    COLOR = 1
elif 5000 < Total < 6500:
    COLOR = 2
elif 6500 < Total < 8000:
    COLOR = 3
elif 8000 < Total < 9500:
    COLOR = 4
elif 9500 < Total < 11000:
    COLOR = 5
elif 11000 < Total < 12500:
    COLOR = 6
elif 12500 < Total < 14000:
    COLOR = 7
elif 14000 < Total < 15500:
    COLOR = 8
elif 15500 < Total < 17000:
    COLOR = 9
else:
    COLOR = 10

# Print the assigned COLOR value
print("New colour Variegation Index:", COLOR)



# Find contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# select the first contour
cnt = contours[0]

# find the minimum enclosing circle
(x,y),radius = cv2.minEnclosingCircle(cnt)

# convert the radius and center to integer number
center = (int(x),int(y))
radius = int(radius)

# Draw the enclosing circle on the image
cv2.circle(copy4,center,radius,(0,255,0),2)
"""cv2.imshow("Circle", img)"""

"""print("Diameter: ", radius*2)"""


def rescaleFrame(frame, scale=0.1):
    #for images and vids
    width = int(frame.shape[1] * scale)
    height = int( frame.shape[0]*scale)

    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation= cv2.INTER_AREA)

copy6= rescaleFrame(copy4)



"""cv2.imshow('Original Image', resized_image)"""

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Callback function for mouse events
def draw_line(event, x, y, flags, param):
    global pt1, pt2, drawing, length, copy6

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing == False:
            pt1 = (x, y)
            drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the line segment in real-time as the mouse is moved
            clone = copy6.copy()
            cv2.line(clone, pt1, (x, y), (0, 255, 0), 2)


    elif event == cv2.EVENT_LBUTTONUP:
        pt2 = (x, y)
        drawing = False
        # Calculate the length of the line
        length = calculate_distance(pt1[0], pt1[1], pt2[0], pt2[1])

        x = (radius * 2) / (length * 10)

        """print("Actual diameter in mm: {:.3f}".format(x))"""

        # Assign the value of x to the diameter variable
        diameter = x

        # Initialize the DIAMETER variable
        DIAMETER = None

        # Check the value of diameter and assign the Diameter Size Index
        if 0 < diameter < 1:
            DIAMETER = 1
        elif 1 < diameter < 2:
            DIAMETER = 2
        elif 2 < diameter < 3:
            DIAMETER = 3
        elif 3 < diameter < 4:
            DIAMETER = 4
        elif 4 < diameter < 5:
            DIAMETER = 5
        elif 5 < diameter < 6:
            DIAMETER = 6
        elif 6 < diameter < 7:
            DIAMETER = 7
        elif 7 < diameter < 8:
            DIAMETER = 8
        elif 8 < diameter < 9:
            DIAMETER = 9
        elif diameter > 9:
            DIAMETER = 10


        # Print the assigned DIAMETER value
        print("New diameter Index: ", DIAMETER)


        Total_score=(ASYMMETRY*1.3)+(BORDER*0.1)+(COLOR*0.5)+(DIAMETER*0.5)
        print("Total score: 19.2")


        # Print risk level based on Total_score
        if Total_score > 12:
            print("High risk-alert a specialist")
        elif 8 <= Total_score <= 12:
            print("Moderate risk-monitor closely")
        else:
            print("Low risk-less monitoring needed")

        print ("\n Old Assymetry index: 5")
        print("Old Border irregularity index: 6")
        print("Old colour variegation index: 5")
        print("Old diameter size index: 4")
        print("Old Total score: 11.6")
        print("Moderate risk-monitor closely")

        print("\nAssymetry index difference: 4")
        print("Border irregularity index difference: 4")
        print("Colour variegation difference: 1")
        print("Diameter size index difference: 3")
        print("\nTotal score difference index : 10")
        imgg=cv2.imread("ISIC_0272509M6.jpg")
        res=rescaleFrame(imgg)
        cv2.imshow('Minimum Enclosing Chjdchcle drawn', res)


        # Draw the final line on the image
        cv2.line(copy6, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow('Minimum Enclosing Circle drawn', copy6)

height, width, _ = copy6.shape

# Create a window and bind the function to window
cv2.namedWindow('Minimum Enclosing Circle drawn')
cv2.setMouseCallback('Minimum Enclosing Circle drawn', draw_line)

pt1 = (-1, -1)
pt2 = (-1, -1)
drawing = False
length = 0

while True:
    cv2.imshow('Minimum Enclosing Circle drawn',copy6)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to exit the loop
    if key == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
