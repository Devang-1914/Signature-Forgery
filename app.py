import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib.pyplot as plt
import imagehash


def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def compare_images(image1, image2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images
    (score, _) = ssim(gray1, gray2, full=True)
    return score


def preprocessing(image):
    # Convert image to grayscale
    gray = image.convert("L")

    # Convert grayscale image to numpy array
    img = np.array(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
    morphology_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Apply median blur
    blur = cv2.GaussianBlur(morphology_img, (3, 3), 0)

    # Apply thresholding
    _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the bounding box coordinates of the non-white pixels
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)

    # Add extra white space to the bounding box coordinates
    padding = 5  # Adjust the padding size as needed
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Make sure the coordinates are within the image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    # Crop the image using the modified bounding box coordinates
    cropped_image = binary[y:y + h, x:x + w]

    # Add extra white space around the cropped image
    extra_space = np.zeros((cropped_image.shape[0] + 2 * padding, cropped_image.shape[1] + 2 * padding),
                           dtype=np.uint8) * 255
    extra_space[padding:-padding, padding:-padding] = cropped_image

    corrected = cv2.resize(extra_space, (330, 175))
    # Convert the numpy array back to PIL image
    resized_image = Image.fromarray(corrected)
    return resized_image


def display_images(img):
    # Assuming 'preprocessing' is a function that modifies the image and returns a PIL image
    after_preprocessing = preprocessing(img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="before_preprocessing")

    with col2:
        st.image(after_preprocessing, caption="after_preprocessing")

def hash_compare(img1, img2):
    # Generate perceptual hashes for each image
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    # Compare the hashes
    # The smaller the difference, the more similar the images are.
    difference = hash1 - hash2

    st.write(f"Hash of the first image: {hash1}")
    st.write(f"Hash of the second image: {hash2}")
    st.write(f"Difference between the hashes: {difference}")

    # Depending on your application, you might define a threshold for
    # considering images to be 'similar'. For example:
    threshold = 5  # This is arbitrary; adjust based on your requirements
    if difference < threshold:
        st.write("The images are similar.")
    else:
        st.write("The images are not similar.")

def main():
    st.title("Signature Verification")

    # Allow user to upload two image files
    i = st.camera_input("Take a picture", key="camera1")
    i_2 = st.camera_input("Take another picture", key="camera2")
    gen = st.file_uploader("Upload Genuine Signature", type=["png", "jpg", "jpeg"])
    forged = st.file_uploader("Upload Forged Signature", type=["png", "jpg", "jpeg"])

    if i_2 and i:
        genuine_signature = cv2.imdecode(np.fromstring(i_2.read(), np.uint8), cv2.IMREAD_COLOR)
        forged_signature = cv2.imdecode(np.fromstring(i.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display the uploaded images in Streamlit
        st.image(Image.open(i_2), caption="Signature 1", use_column_width=True)
        st.image(Image.open(i), caption="Signature 2", use_column_width=True)

        # Resize both images to a common size for comparison
        common_width = 200
        common_height = 100
        genuine_signature_resized = resize_image(genuine_signature, common_width, common_height)
        forged_signature_resized = resize_image(forged_signature, common_width, common_height)
        #pre_gen = preprocessing(Image.open(i_2))
        #pre_for = preprocessing(Image.open(i))
        display_images(Image.open(i_2))
        display_images(Image.open(i))



        # Compare the resized images using SSIM
        similarity_score = compare_images(genuine_signature_resized, forged_signature_resized)
        #similarity_score = compare_images(pre_gen, pre_for)


        # Set a threshold to classify real and fake signatures
        threshold = 0.9

        # Display the result
        if similarity_score > threshold:
            st.write("The signature is Genuine.")
            st.write(similarity_score)
        else:
            st.write("The signature is fake.")
            st.write(similarity_score)

    if gen and forged:
        genuine_signature = cv2.imdecode(np.fromstring(gen.read(), np.uint8), cv2.IMREAD_COLOR)
        forged_signature = cv2.imdecode(np.fromstring(forged.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display the uploaded images in Streamlit
        st.image(Image.open(gen), caption="Signature 1", use_column_width=True)
        st.image(Image.open(forged), caption="Signature 2", use_column_width=True)

        # Resize both images to a common size for comparison
        common_width = 200
        common_height = 100
        genuine_signature_resized = resize_image(genuine_signature, common_width, common_height)
        forged_signature_resized = resize_image(forged_signature, common_width, common_height)
        #pre_gen = preprocessing(Image.open(i_2))
        #pre_for = preprocessing(Image.open(i))
        display_images(Image.open(gen))
        display_images(Image.open(forged))



        # Compare the resized images using SSIM
        similarity_score = compare_images(genuine_signature_resized, forged_signature_resized)
        #similarity_score = compare_images(pre_gen, pre_for)



        # Set a threshold to classify real and fake signatures
        threshold = 0.9

        # Display the result
        if similarity_score > threshold:
            st.write("The signature is Genuine.")
            st.write(similarity_score)
        else:
            st.write("The signature is fake.")
            st.write(similarity_score)

        hash_compare(Image.open(gen), Image.open(forged))

if __name__ == "__main__":
    main()
