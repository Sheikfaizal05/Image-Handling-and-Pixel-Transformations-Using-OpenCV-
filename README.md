# Image Handling and Pixel Transformations Using OpenCV

## Aim

Use OpenCV and Matplotlib to read, display, save, crop, resize, flip, and annotate images; adjust image brightness and contrast; and inspect and merge BGR and HSV color channels.

## Software Required

- Python 3.7 or later
- Anaconda (or another Python environment)
- Jupyter Notebook
- OpenCV (`opencv-python`), NumPy, and Matplotlib

## Exercise 01

**Name:** AKASH CT  
**Register No.:** 212224240007  
**Slot No.:** T2  
**Slot Code:** 26OD2199  
**Academic Year:** 2026-27

Place `Eagle_in_Flight.jpg`, `Apollo-11-launch.jpg`, and `boy.jpg` in the notebook's working directory before running the cells.

### 1. Import the required libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Read and display the eagle image in grayscale

```python
faceImage = cv2.imread("Eagle_in_Flight.jpg", cv2.IMREAD_GRAYSCALE)
height, width = faceImage.shape
print("Width:", width)
print("Height:", height)
print("Channel:", 1)

plt.imshow(faceImage, cmap="gray")
plt.show()
```

### 3. Save the grayscale image and convert it to color

```python
cv2.imwrite("Eagle_new.png", faceImage)
img_color = cv2.cvtColor(faceImage, cv2.COLOR_GRAY2RGB)

height, width, channel = img_color.shape
print("Width:", width)
print("Height:", height)
print("Channel:", channel)

plt.imshow(img_color)
plt.show()
```

### 4. Crop, resize, and flip the eagle

```python
eagle_crop = img_color[300:400, 300:550]
plt.imshow(eagle_crop)
plt.title("Cropped Eagle")
plt.show()

eagle_resized = cv2.resize(eagle_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
plt.imshow(eagle_resized)
plt.title("Eagle Resized 2x")
plt.show()

eagle_flipped = cv2.flip(eagle_resized, 1)
plt.imshow(eagle_flipped)
plt.title("Eagle Flipped Horizontally")
plt.show()
```

### 5. Annotate the Apollo 11 launch image

```python
apollo = cv2.imread("Apollo-11-launch.jpg")

text = "Apollo 11 Saturn V Launch, July 16, 1969"
font_face = cv2.FONT_HERSHEY_PLAIN
font_scale = 1.5
thickness = 2
text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
text_x = (apollo.shape[1] - text_size[0]) // 2
text_y = apollo.shape[0] - 20

cv2.putText(
    apollo, text, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness
)

magenta = (255, 0, 255)  # BGR color
cv2.rectangle(apollo, (100, 50), (450, 800), magenta, 3)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(apollo, cv2.COLOR_BGR2RGB))
plt.title("Apollo 11 Launch - Annotated")
plt.axis("on")
plt.show()
```

### 6. Read the boy image and adjust brightness

```python
img = cv2.imread("boy.jpg")

matrix_ones = np.ones(img.shape, dtype=np.float64)
matrix = (matrix_ones * 50).astype(np.uint8)

img_brighter = cv2.add(img, matrix)
img_darker = cv2.subtract(img, matrix)

plt.figure(figsize=(12, 4))
for index, (image, title) in enumerate(
    [
        (img, "Original Image"),
        (img_darker, "Darker Image"),
        (img_brighter, "Brighter Image"),
    ]
):
    plt.subplot(1, 3, index + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
```

### 7. Modify image contrast

```python
matrix1 = np.ones(img.shape, dtype=np.float64)
matrix2 = np.ones(img.shape, dtype=np.float64)

img_lower = cv2.multiply(img, matrix1, scale=0.8, dtype=cv2.CV_8U)
img_higher1 = cv2.multiply(img, matrix1, scale=1.1, dtype=cv2.CV_8U)
img_higher2 = cv2.multiply(img, matrix2, scale=1.2, dtype=cv2.CV_8U)

plt.figure(figsize=(14, 4))
images = [
    (img, "Original"),
    (img_lower, "Lower Contrast (0.8)"),
    (img_higher1, "Higher Contrast (1.1)"),
    (img_higher2, "Higher Contrast (1.2)"),
]

for index, (image, title) in enumerate(images):
    plt.subplot(1, 4, index + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
```

### 8. Split and merge BGR channels

```python
b, g, r = cv2.split(img)

plt.figure(figsize=(12, 4))
for index, (channel, title) in enumerate(
    [
        (b, "Blue Channel"),
        (g, "Green Channel"),
        (r, "Red Channel"),
    ]
):
    plt.subplot(1, 3, index + 1)
    plt.imshow(channel, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

merged_bgr = cv2.merge([b, g, r])
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB))
plt.title("Merged BGR Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.tight_layout()
plt.show()
```

### 9. Split and merge HSV channels

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

plt.figure(figsize=(12, 4))
for index, (channel, title) in enumerate(
    [
        (h, "Hue Channel"),
        (s, "Saturation Channel"),
        (v, "Value Channel"),
    ]
):
    plt.subplot(1, 3, index + 1)
    plt.imshow(channel, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

merged_hsv = cv2.merge([h, s, v])
merged_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(merged_rgb)
plt.title("Merged HSV Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.tight_layout()
plt.show()
```

## Output

- Read, displayed, and saved the eagle image; cropped, resized, and flipped the selected region.
- Added a caption and magenta rectangle to the Apollo 11 launch image.
- Compared original, brighter, darker, and contrast-adjusted versions of the boy image.
- Displayed and reconstructed the image using its BGR and HSV channels.

## Result

The images were successfully processed and displayed using OpenCV, NumPy, and Matplotlib. Brightness and contrast were adjusted, and BGR and HSV channels were split and merged.

