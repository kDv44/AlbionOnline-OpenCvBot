import cv2 as cv
import numpy as num


stack_img = cv.imread("source_image\\albion_farm.jpg", cv.IMREAD_REDUCED_COLOR_2)
needle_img = cv.imread("source_image\\albion_cabbage.jpg", cv.IMREAD_REDUCED_COLOR_2)

result = cv.matchTemplate(stack_img, needle_img, cv.TM_CCOEFF_NORMED)


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
    #cv.imshow('cabbage_result', stack_img)
    #cv.waitKey()
    cv.imwrite('source_image\\cabbage_result.jpg', stack_img)

else:
    print("Needle not found.")
