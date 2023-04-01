import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'



original_image = cv2.imread('testing_images\\car7.jpg')

original_image = imutils.resize(original_image, width=500 )
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 
gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)


edged_image = cv2.Canny(gray_image, 30, 200)

contours, new = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = original_image.copy()
cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
cv2.imshow("img1", img1)


contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]


screenCnt = None
img2 = original_image.copy()

# Visualilze the top 30 contours
cv2.drawContours(img2, contours, -1, (0, 255, 0), 3)
cv2.imshow("img2", img2)


for c in contours:
	contour_perimeter = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.018 * contour_perimeter, True)

	if len(approx) == 4:
		screenCnt = approx

		x, y, w, h = cv2.boundingRect(c)
		new_img = original_image[y : y+h, x : x+w]

		cv2.imwrite('./' + 'cropped' + '.png', new_img)
		break



# creates a red boundary around the license plate contour on original image
cv2.drawContours(original_image , [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("detected license plate", original_image )


# filename of the cropped license plate image
cropped_License_Plate = './cropped.png' 
cv2.imshow("cropped license plate", cv2.imread(cropped_License_Plate))


# converts the license plate characters to string
text = pytesseract.image_to_string(cropped_License_Plate) 


print("License plate is:", text)
cv2.waitKey(0)
cv2.destroyAllWindows()