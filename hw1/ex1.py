import cv2 
import matplotlib.pyplot as plt
import numpy as np

# Load an image from file as function
def load_image(image_path):
    image = cv2.imread(image_path)
    # need to change colors!!!
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

# Display an image as function
def display_image(image, title="Image"):
    plt.imshow(image)
    plt.title(title)


# grayscale an image as function
def grayscale_image(image):
    height = image.shape[0]
    width = image.shape[1]

    img_gray = np.zeros((height, width))

    weights = np.array([0.2989, 0.5870, 0.1140])

    for i in range(height):
        for j in range(width):
            orig = image[i, j]
            gray = np.multiply(orig, weights)
            img_gray[i,j] = np.sum(gray)

    return img_gray


# Save an image as function
def save_image(image, output_path):
    cv2.imwrite(output_path, image) 



# flip an image as function 
def flip_image(image):
    res = cv2.flip(image, 1) 
    return res


# rotate an image as function
def rotate_image(image, angle):
    height = image.shape[0]
    width = image.shape[1]
    center = (width/2, height/2)
    size = (width, height)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    res = cv2.warpAffine(image, rot_mat, size)
    return res


if __name__ == "__main__":
    # Load an image from file
    img = load_image("images/uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")
    # save_image(img_gray_flipped, "images/lena_gray_flipped.jpg")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/lena_gray_rotated.jpg")

    # Show the images
    plt.show() 
