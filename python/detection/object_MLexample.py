import cv2

frame = cv2.imread("../../input/debris_image2.jpeg")

#cv2.rectangle(frame, (535, 190), (835, 320), (0, 0, 255), 4)
cv2.putText(frame, "Detected object:", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.putText(frame, "Chair, 90.030", (290, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

while True:
    cv2.imshow("Object in waterway classified", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()