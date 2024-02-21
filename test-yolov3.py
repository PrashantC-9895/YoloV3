import numpy as np
import cv2

'''
resizing the image if image size is greater than 750*750 pixels 
'''
def resize_image(image_path):
    image = cv2.imread(image_path)
    print(f'shape of Original image : {image.shape}')

    width , height = image.shape[1] , image.shape[0] #taking orginal image width and height

    if width > 750 or height > 750:

        resized_image = cv2.resize(image , (650 , 500)) #resizing the image
        print(f'Resized image shape : {resized_image.shape}')
        return resized_image

    else:
        return image

#step-1
#image as an input
new_image = resize_image('two_bikes.jpg')


image_width , image_height = new_image.shape[1], new_image.shape[0]
print(f'original image width: {image_width}')
print(f'original image height : {image_height}')

'''
reading the classse names form txt file 
'''
class_names = []
k = open('ms_coco_classnames.txt','r')
for i in k.readlines():
    class_names.append(i.strip())
print(class_names)

'''
Reading the configuration file and weights file
'''
Neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights') ##dnn -> deep neural network , laoding YOLOv3 configuration and weights files

#setup the inpute image to architecutre
blob = cv2.dnn.blobFromImage(new_image, 1/255 , (320,320), True , crop = False) #blob is a pre-processed image that serves as input to a deep neural network
print(f'input for the model :{blob.shape}')

#sending input to the model
Neural_network.setInput(blob) #sets the input blob for the neural network

#getting output layers of YolovV3
cfg_data = Neural_network.getLayerNames()
layers_names = Neural_network.getUnconnectedOutLayers()
print(layers_names)

outputs = [cfg_data [i-1] for i in layers_names]
print(outputs)

output_data = Neural_network.forward(outputs)
print(output_data[0].shape)
print(output_data[1].shape)
print(output_data[2].shape)

#step-2
Threshold = 0.80
image_size = 320


# Iterate through the output data to extract bounding box coordinates, class IDs, and confidence scores
def bounding_box(output_data):
    cordinates = []
    ids = []
    confidance_score = []
    for i in output_data:
        for j in i:
            # Extract class probabilities from the output data
            prob_values = j[5:]
            # Determine the class with the highest probability
            class_ = np.argmax(prob_values)
            # Retrieve the confidence score of the detected object
            confidance = prob_values[class_]
            # Check if the confidence score exceeds the threshold
            if confidance > Threshold:

                # Calculate the width and height of the bounding box
                w , h = int(j[2] * image_size) , int(j[3] * image_size)
                # Calculate the coordinates of the bounding box
                x , y = int(j[0] * image_size - w / 2) , int(j[1] * image_size - h / 2)

                # Append the bounding box coordinates to the list
                cordinates.append([x,y,w,h])
                # Append the class ID to the list
                ids.append(class_)
                # Append the confidence score to the list
                confidance_score.append(confidance)



    print(cordinates)
    print(confidance_score)
    print(ids)

    final_box = cv2.dnn.NMSBoxes(cordinates , confidance_score , Threshold , .65)
    return final_box , cordinates , confidance_score , ids


#step - 3
def predict_(final_box , cordinates , confidance_score , ids , width_ratio , height_ratio):
    # Print the final bounding box
    print(final_box)
    # Check if there are any bounding boxes detected
    if len(final_box) != 0 :
        # Iterate through each bounding box
        for i in final_box.flatten():

            # Extract coordinates of the bounding box
            x,y,w,h = cordinates[i]

            # Resize the coordinates based on the width and height ratio
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)

            # Define font type
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Convert confidence score to string
            f = str(round(confidance_score[i],2))
            # Generate text with class name and confidence score
            text = str(class_names[ids[i]]) + '-' + f

            # Draw bounding box on the image
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            # Add text label to the bounding box
            cv2.putText(new_image ,text, (x, y-5), font , 1 , (255, 255, 255), 2, cv2.LINE_AA)

    else:
        print('no object detected')


final_box , cordinates , confidance_score , ids = bounding_box(output_data)
predict_(final_box , cordinates , confidance_score , ids , image_width / 320 , image_height / 320)



cv2.imshow('im', new_image)
cv2.waitKey()
cv2.destroyAllWindows()