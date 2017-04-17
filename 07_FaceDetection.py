import cv2
print(cv2.__version__)
import os.path
import urllib.request



video = 'ixyYIA_wj-U'

videoYid=[]
number_faces=[]

urllib.request.urlretrieve("http://img.youtube.com/vi/" + video + "/0.jpg", video + ".jpg")
faceCascade = cv2.CascadeClassifier("./source/haarcascade_frontalface_default.xml")
image = cv2.imread(video + ".jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE
)

videoYid.append(video)

number_faces.append(len(faces))

print("Found {0} faces!".format(len(faces)))
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	center_point = (int((x + (x + w)) / 2), int((y + (y + h)) / 2))
	cv2.rectangle(image, center_point, center_point, (255, 0, 0), 2)

cv2.imshow("Faces found", image)     #show face in the image
cv2.waitKey(0)
os.remove(video + ".jpg")


