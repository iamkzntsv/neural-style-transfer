import tensorflow as tf


# CONTENT COST
def compute_content_cost(a_C, a_G):
    """
    Compute the content cost (how different two activations are)

    Parameters:
    ----------
    a_C : tensor of shape (1, m, n_H, n_W, n_C)
        hidden layer activations representing content of the image C.
    a_G : tensor of shape (1, m, n_H, n_W, n_C)
        hidden layer activations representing content of the image G.

    Returns:
    -------
    J_content : scalar
        computed cost.
    """
    # Retrieve (m, n_H, n_W, n_C) shape tensors (output from the content computing layer)
    a_C = a_C[-1]
    a_G = a_G[-1]

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.shape

    # Compute cost
    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(a_C - a_G))

    return J_content


# STYLE COST
def gram_matrix(A):
    """
    Compute gram matrix

    Parameters:
    ----------
    A : tensor of shape (n_C, n_H*n_W)
        a matrix of feature vectors from layer activation.

    Returns:
    -------
    GA : tensor of shape (n_C, n_C)
        gram matrix of A.
    """
    GA = tf.linalg.matmul(A, tf.transpose(A))

    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Compute the style cost for a single layer (how different two correlation maps are).

    Parameters:
    ----------
    a_S : tensor of shape (m, n_H, n_W, n_C)
        hidden layer activations representing content of the image S.
    a_G : tensor of shape (m, n_H, n_W, n_C)
        hidden layer activations representing content of the image G.

    Returns:
    -------
    J_style : scalar
        computed cost.
    """
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images from (n_H * n_W, n_C) to (n_C, n_H * n_W)
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))

    # Computing gram_matrices for both images
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Compute the loss
    J_style_layer = tf.reduce_sum(tf.square(GS - GG)) / (4 * (n_C * n_H * n_W) ** 2)

    return J_style_layer


def compute_style_cost(style_image_output, generated_image_output, style_layers):
    """
    Compute overall style cost.

    Parameters:
    ----------
    style_image_output :
    generated_image_output :
    style_layers : a python list containing:
                      - the names of the layers we would like to extract style from
                      - a coefficient for each of them

    Returns:
    -------
    J_style : tensor representing a scalar value
       style cost over all layers.
    """

    # Initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]

    for i, weight in zip(range(len(a_S)), style_layers):
        # Compute style cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


# TOTAL COST
@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Compute the total cost

    Parameters:
    ----------
    J_content : a scalar
         content cost
    J_style : a scalar
         style cost
    alpha : a scalar
         hyperparameter weighting the importance of the content cost
    beta : a scalar
         hyperparameter weighting the importance of the style cost

    Returns:
    -------
    J : a scalar
        total cost.
    """
    J = alpha * J_content + beta * J_style
    return J
