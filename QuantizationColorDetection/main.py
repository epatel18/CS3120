from sklearn.cluster import KMeans as km
from collections import Counter as count
import matplotlib.pyplot as plt
import cv2
import csv
import numpy as np

# Retrieves a sample image from the path and converts the default OpenCV BGR value tto RGB value
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Formats colors to hexadecimal values that are padded up to 2 places
def hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

# Image is resized to get colors faster, KMeans is implemented to get first fit, and colors are ordered together by clusters
def get_colors(image, color_num):
    img = cv2.resize(image, (800, 600))
    img = img.reshape(img.shape[0] * img.shape[1], 3)

    cluster = km(n_clusters = color_num)
    labels = cluster.fit_predict(img)
    ct = count(labels)

    center = cluster.cluster_centers_
    order = [center[i] for i in ct.keys()]
    hex_color = [hex(order[i]) for i in ct.keys()]

    # Prints hexadecimal value of colors detected to the console
    for color in hex_color:
        print(f"[*] {color}")
    
    # Prompts for if user wants data representation of colors and their frequency.
    chart = input("\n>> Would you like a pie chart visualization? [y/n] ")
    if chart.lower() == "y":
       plt.figure(figsize = (10, 8))
       plt.pie(ct.values(), labels = hex_color, colors = hex_color)
       plt.show()

# retrieves image with quantinized colors using the KMeans algorithm
def quantinization(img, k):
    data = np.float32(img).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    quant = center[label.flatten()]
    quant = quant.reshape(img.shape)
    return quant

# main method to prompt for user input and plot the data
def main():
    path = input(">> Please enter the image path: ")
    color_num = int(input(">> Please enter the number of colors you would like to find: "))
    print()
    
    try:
        image = cv2.imread(path)
        picture = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = get_image(path)
        get_colors(img, color_num)
        plt.imshow(quantinization(image,5))
        plt.show()
        plt.imshow(quantinization(image,10))
        plt.show()
        plt.imshow(quantinization(image,50))
        plt.show()
        plt.imshow(quantinization(image,100))
        plt.show()
    except:
        print("Something went wrong with finding colors! Please enter a valid path or format for the image.")
        exit()
    
if __name__ == "__main__":
    main()
