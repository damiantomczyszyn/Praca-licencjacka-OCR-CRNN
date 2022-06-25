# import the necessary packages
import pytesseract
import argparse
import cv2
import string
# construct the argument parser and parse the arguments}
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
args = vars(ap.parse_args())


punclist='.?,()":'





# load the input image and convert it from BGR to RGB channel
# ordering}
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(image)
#text = pytesseract.image_to_string(image, config="-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz")
print(pytesseract.get_tesseract_version())
print(text, end="")

with open('text.txt', 'a') as f:
    f.write(text)


