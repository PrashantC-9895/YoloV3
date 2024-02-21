import numpy as np
import cv2

Threshold = 0.5
image_size = 320

def prediction(final_box, confidance_score , cordinates , ids , width_ratio , height_ratio, image):
    """
    Predicts and visualizes bounding boxes on the input image.

    Args:
        final_box (numpy.ndarray): Array of final bounding box indices.
        confidance_score (list): List of confidence scores for each detected object.
        cordinates (list): List of bounding box coordinates.
        ids (list): List of class IDs for each detected object.
        width_ratio (float): Width ratio used for resizing bounding box coordinates.
        height_ratio (float): Height ratio used for resizing bounding box coordinates.
        image (numpy.ndarray): Input image to draw bounding boxes on.

    Returns:
        str: Message indicating the detection process status.
    """
    if len(final_box) != 0:
        for i in final_box.flatten():
            x, y, w, h = cordinates[i]
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)
            font = cv2.FONT_HERSHEY_PLAIN
            cnf = str(round(confidance_score[i], 2))
            text = str(class_names[ids[i]]) + '--' + cnf
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, text, (x, y - 2), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return 'Detections are in progress.'
    else:
        print('No object to detect in the input image.')

def bounding_box(output_data):
    """
    Extracts bounding box information from YOLOv3 detections.

    Args:
        output_data (numpy.ndarray): YOLOv3 output detections.

    Returns:
        tuple: Final bounding box indices, confidence scores, bounding box coordinates, and class IDs.
    """
    cordinates = []
    ids = []
    confidance_score = []
    for i in output_data:
        for j in i:
            prob_values = j[5:]
            class_ = np.argmax(prob_values)
            confidance = prob_values[class_]
            if confidance > Threshold:
                w , h = int(j[2] * image_size) , int(j[3] * image_size)
                x , y = int(j[0] * image_size - w/2) , int(j[1] * image_size - h/2)
                cordinates.append([x,y,w,h])
                ids.append(class_)
                confidance_score.append(confidance)
    final_box = cv2.dnn.NMSBoxes(cordinates ,confidance_score ,Threshold, 0.60)
    return final_box , confidance_score , cordinates , ids

# Reading video file
cap = cv2.VideoCapture('video - 1.mp4')

# Loading YOLOv3 model
Neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')

# Reading class names from text file
class_names = []
k = open('ms_coco_classnames.txt','r')
for i in k.readlines():
    class_names.append(i.strip())

while cap.read():
    res , frame = cap.read()
    if res == True:
        original_width, original_height = frame.shape[1], frame.shape[0]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), True, crop=False) # Setting input data
        Neural_network.setInput(blob) # Sending input to the model
        cfg = Neural_network.getLayerNames()
        layes_names = Neural_network.getUnconnectedOutLayers()
        outputs = [cfg[i - 1] for i in layes_names]
        output_data = Neural_network.forward(outputs)
        final_box, confidance_score , cordinates , ids = bounding_box(output_data)
        prediction(final_box, confidance_score , cordinates , ids , original_width/320 , original_height/320 , frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
