IMG_SIZE = (400, 400)

# LOAD IMAGES
def load_img_from_url(url):
    try:
        f = urllib.request.urlopen(url)  # load image
        jpeg_str = f.read()
        pic = Image.open(BytesIO(jpeg_str)).resize(IMG_SIZE)  # decode to PIL Image and resize
        img = np.array(pic)  # convert to numpy format
        return img
    except IOError:
        print("Cannot retrieve the image. Please check the url.")
        return


def preprocess_img(_img):
    # Scale pixel values to 0-1 range
    _img = _img.astype(np.float32) / 255
    # Lower resolution
    _img = cv2.resize(_img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
    return _img


# Load content image
content_image = load_img_from_url(
    'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQhPD9kMFrpy6Qv5UjjIQ93SWWrNjSMqngQAQ&usqp=CAU')
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
print(f"Content image shape: {content_image.shape}")

# Load style image
style_image = load_img_from_url(
    'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYTYDW1Lw9Zp3qoO4iScTtWkulBo1KYMbD0A&usqp=CAU')
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
print(f"Style image shape: {style_image.shape}")

# Initialize the generated image as a noisy image slightly corelated with the content image,
# this will help the content of both images to match more rapidly.
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
generated_image = tf.Variable(generated_image)

# Show the 3 images in a row
fig = plt.figure(figsize=(16, 4))

ax = fig.add_subplot(1, 3, 1)
plt.imshow(content_image[0])
ax.title.set_text('Content image')

ax = fig.add_subplot(1, 3, 2)
plt.imshow(style_image[0])
ax.title.set_text('Style image')

ax = fig.add_subplot(1, 3, 3)
plt.imshow(generated_image[0])
ax.title.set_text('Generated image')

plt.savefig("inital.png")
plt.show()