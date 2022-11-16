import matplotlib.pyplot as plt
from model.vgg import NST
from data_loader import content_image, style_image, generated_image
from loss import compute_content_cost, compute_style_cost, total_cost
from utils import *

# Define layer for computing content cost and its weight
CONTENT_LAYER = [('block5_conv4', 1)]

# Define layers and their weights for computing style cost
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

# Build the model
model = NST(STYLE_LAYERS + CONTENT_LAYER).build()

# Preprocessing
preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = model(preprocessed_content)

preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = model(preprocessed_style)

# Training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


@tf.function()
def train_step(G):
    """
    Perform a single training step.

    Parameters:
    ----------
    G : tensor
        generated image.

    Returns:
    -------
    total cost.
    """

    with tf.GradientTape() as tape:
        # Compute the model output for current generated image
        a_G = model(G)

        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)

        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS)

        # Compute the total cost
        J = total_cost(J_content, J_style)

    # Compute gradients
    grad = tape.gradient(J, G)

    # Update a_G
    optimizer.apply_gradients([(grad, G)])
    G.assign(clip_0_1(G))

    return J


# Main loop
epochs = 20000
for i in range(epochs):
    train_step(generated_image)
    if i % 250 == 0:
        print(f"Epoch {i} ")
        image = tensor_to_image(generated_image)
        plt.axis('off')
        plt.imshow(image)
        plt.savefig("figures/img" + str(i) + ".png")
