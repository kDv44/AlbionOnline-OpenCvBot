import cv2 as cv
import numpy as np
import sys


np.set_printoptions(threshold=sys.maxsize)

stack_img = cv.imread("source_image\\albion_farm.jpg", cv.IMREAD_UNCHANGED)
needle_img = cv.imread("source_image\\albion_cabbage.jpg", cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(stack_img, needle_img, cv.TM_SQDIFF_NORMED)
print(result)

threshold = 0.17
locations = np.where(result <= threshold)
locations = list(zip(*locations[::-1]))
print(locations)

if locations:
    print("Found needle.")

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
        cv.rectangle(stack_img, top_left, bottom_right, line_color, line_type)

    cv.imshow('Matches', stack_img)
    cv.waitKey()

else:
    print("Needle not found.")


'''
# don't touch this (first_copper)

# get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print("Best match top left position: %s" % str(max_loc))
print("Best match confidence: %s" % max_val)

threshold = 0.8
if max_val >= threshold:
    print("Found needle.")

    # get dimesions the needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    cv.rectangle(
        stack_img, top_left, bottom_right,
        color=(0, 255, 0), thickness=2, lineType=cv.LINE_4,
    )
    # this for looks
    # cv.imshow('cabbage_result', stack_img)
    # cv.waitKey()
    cv.imwrite('source_image\\cabbage_result.jpg', stack_img)

else:
    print("Needle not found.")
'''