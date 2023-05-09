import cv2

image = cv2.imread("../../input/human_waterway_example.jpeg")
box_x = 600
box_width = 820
box_y = 140
box_height = 230
cv2.rectangle(image, (box_x, box_y), (box_width, box_height), (0,0,255))
cv2.rectangle(image, (int(box_x),int(box_y)), (int(box_width),int(box_y+35)), (0,0,255), -1) #int((box_y+((box_height-box_y)/7)))
cv2.putText(image, "Person 86.1", (int(box_x), int(box_y + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
cv2.putText(image, "Human detected", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.circle(image, (int(box_x+(box_width-box_x)/2), int(box_y+(box_height-box_y)/2)), 7, (0,0,255), -1)
cv2.putText(image, "32.68 FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

while True:
    cv2.imshow("Person Detection", image)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()