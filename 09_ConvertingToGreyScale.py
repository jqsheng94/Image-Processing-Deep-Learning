import cv2
import matplotlib.pyplot as plt


#Convert to Grey Image

bgr_img = cv2.imread('./inception/san_francisco.jpg')
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./inception/san_francisco_grayscale.jpg',gray_img)

plt.imshow(gray_img, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()



#Convert to Blue and Red


bgr_img = cv2.imread('./inception/san_francisco.jpg')
b,g,r = cv2.split(bgr_img)       # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb
cv2.imwrite('./inception/san_francisco_yellowscale.jpg',bgr_img)


plt.imshow(rgb_img)
plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
plt.show()




