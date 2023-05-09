import cv2

frame = cv2.imread("../../input/debris_image2.jpeg")

cv2.rectangle(frame, (535, 190), (835, 320), (0, 0, 255), 4)
cv2.putText(frame, "-------------------------------", (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
cv2.putText(frame, "-------------------------------", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)
cv2.putText(frame, "Object", (535, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(frame, "Detection below this line", (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame, "Detection above this line", (15, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

while True:
    cv2.imshow("Object detected in waterway", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()