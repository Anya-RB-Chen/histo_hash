import os
import cv2
import pandas as pd

# Set the path to the root directory of the Stanford Dogs Dataset
root_dir = 'dataset/StanfordDogsDataset'

# Initialize a list to store the image data
data = []

# Traverse all folders in the dataset
for dir_name, _, file_names in os.walk(root_dir):
    count = 0
    for file_name in file_names:
        # Check if the file is an image
        if file_name.endswith('.jpg') and count < 10:
            # Construct the full path to the image file
            file_path = os.path.join(dir_name, file_name)

            # Load the image and convert it to grayscale
            image = cv2.imread(file_path)
            image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Flatten the 2D grayscale image into a 1D array
            image_data = image.flatten()

            # Append the image data to the list
            data.append(image_data)
            print('Processed', file_path, '...', image_data.shape)
            count += 1
        if count == 10:
            break

# Convert the list of image data into a DataFrame
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
df.to_csv('stanford_dogs.csv', index=False)

