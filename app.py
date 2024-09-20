import streamlit as st
import cv2
import numpy as np
from skimage.util import random_noise
from skimage import img_as_ubyte
from PIL import Image, ImageEnhance
import scipy.ndimage

# Title and Instructions
st.title("Coruppting Images")
st.write("Visualizing Image corruption with specified params")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # Display original image
    st.subheader("Original Image")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Corruption Type Selection
    st.subheader("Corruption Type:")

    corruption_options = [
        "None",
        "Gaussian Noise",
        "Shot Noise",
        "Impulse Noise",
        "Defocus Blur",
        "Frosted Glass Blur",
        "Motion Blur",
        "Zoom Blur",
        "Snow",
        "Frost",
        "Fog",
        "Brightness",
        "Contrast",
        "Elastic Distortion",
        "Pixelate",
        "JPEG Compression"
    ]

    corruption_type = st.selectbox("Corruption Type", corruption_options)

    # Corruption Parameters
    if corruption_type == "Gaussian Noise":
        noise_var = st.slider("Variance", 0.01, 0.1, 0.05)
    elif corruption_type == "Shot Noise":
        shot_amount = st.slider("Amount", 0.01, 0.5, 0.1)
    elif corruption_type == "Impulse Noise":
        impulse_amount = st.slider("Amount", 0.01, 0.5, 0.1)
    elif corruption_type == "Defocus Blur":
        kernel_size = st.slider("Kernel Size", 3, 15, 5, step=2)
    elif corruption_type == "Motion Blur":
        motion_degree = st.slider("Degree of Motion Blur", 1, 15, 5)
    elif corruption_type == "Zoom Blur":
        zoom_amount = st.slider("Zoom Amount", 0.01, 2.0, 0.2)
    elif corruption_type == "Brightness":
        brightness_factor = st.slider("Brightness Factor", 0.1, 3.0, 1.0)
    elif corruption_type == "Contrast":
        contrast_factor = st.slider("Contrast Factor", 0.1, 3.0, 1.0)
    elif corruption_type == "Elastic Distortion":
        alpha = st.slider("Alpha (Intensity)", 1, 10, 5)
        sigma = st.slider("Sigma (Elasticity)", 0.5, 2.0, 1.0)
    elif corruption_type == "Pixelate":
        pixelation_size = st.slider("Pixelation Block Size", 2, 20, 10)
    elif corruption_type == "JPEG Compression":
        jpeg_quality = st.slider("JPEG Quality", 10, 100, 50)

    # Corruption Functions
    def apply_gaussian_noise(img, var):
        return img_as_ubyte(random_noise(img, mode='gaussian', var=var))

    def apply_shot_noise(img, amount):
        return img_as_ubyte(random_noise(img, mode='poisson', clip=True))

    def apply_impulse_noise(img, amount):
        return img_as_ubyte(random_noise(img, mode='s&p', amount=amount))

    def apply_defocus_blur(img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def apply_frosted_glass_blur(img):
        return cv2.medianBlur(img, 7)

    def apply_motion_blur(img, degree):
        kernel = np.zeros((degree, degree))
        kernel[int((degree - 1) / 2), :] = np.ones(degree)
        kernel /= degree
        return cv2.filter2D(img, -1, kernel)

    def apply_zoom_blur(img, zoom_amount):
        return scipy.ndimage.zoom(img, (zoom_amount, zoom_amount, 1))

    def apply_snow(img):
        snow_layer = random_noise(img, mode='salt', amount=0.02)
        return img_as_ubyte(snow_layer)

    def apply_frost(img):
        frost_layer = np.random.randn(*img.shape) * 50
        return np.clip(img + frost_layer, 0, 255).astype(np.uint8)

    def apply_fog(img):
        fog_layer = cv2.blur(img, (21, 21))
        return fog_layer

    def apply_brightness(img, factor):
        enhancer = ImageEnhance.Brightness(Image.fromarray(img))
        return np.array(enhancer.enhance(factor))

    def apply_contrast(img, factor):
        enhancer = ImageEnhance.Contrast(Image.fromarray(img))
        return np.array(enhancer.enhance(factor))

    def apply_elastic_distortion(img, alpha, sigma):
        random_state = np.random.RandomState(None)
        shape = img.shape
        dx = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y, _ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(np.zeros_like(dx), (-1, 1))
        return scipy.ndimage.map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)

    def apply_pixelate(img, block_size):
        small_img = cv2.resize(img, (img.shape[1] // block_size, img.shape[0] // block_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    def apply_jpeg_compression(img, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)


    def apply_corruption(img, corruption_type):
        if corruption_type == "Gaussian Noise":
            return apply_gaussian_noise(img, noise_var)
        elif corruption_type == "Shot Noise":
            return apply_shot_noise(img, shot_amount)
        elif corruption_type == "Impulse Noise":
            return apply_impulse_noise(img, impulse_amount)
        elif corruption_type == "Defocus Blur":
            return apply_defocus_blur(img, kernel_size)
        elif corruption_type == "Frosted Glass Blur":
            return apply_frosted_glass_blur(img)
        elif corruption_type == "Motion Blur":
            return apply_motion_blur(img, motion_degree)
        elif corruption_type == "Zoom Blur":
            return apply_zoom_blur(img, zoom_amount)
        elif corruption_type == "Snow":
            return apply_snow(img)
        elif corruption_type == "Frost":
            return apply_frost(img)
        elif corruption_type == "Fog":
            return apply_fog(img)
        elif corruption_type == "Brightness":
            return apply_brightness(img, brightness_factor)
        elif corruption_type == "Contrast":
            return apply_contrast(img, contrast_factor)
        elif corruption_type == "Elastic Distortion":
            return apply_elastic_distortion(img, alpha, sigma)
        elif corruption_type == "Pixelate":
            return apply_pixelate(img, pixelation_size)
        elif corruption_type == "JPEG Compression":
            return apply_jpeg_compression(img, jpeg_quality)
        else:
            return img


    if corruption_type != "None":
        corrupted_image = apply_corruption(img_np, corruption_type)
        st.subheader("Corrupted Image")
        st.image(corrupted_image, caption='Corrupted Image', use_column_width=True)