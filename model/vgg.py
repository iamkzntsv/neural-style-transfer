import tensorflow as tf

IMG_SIZE = (400, 400)
INPUT_SHAPE = IMG_SIZE + (3,)


class NST:

    def __init__(self, layer_names):
        self.layer_names = layer_names

    def build(self):
        """
        Create a vgg model that will return a list of intermediate outputs values for the middle layers

        Parameters:
        ----------
        vgg : keras model
            trained vgg model
        layer_names : a list of tuples where:
            - the first element is the layer name
            - the second element is the weight for that layer

        Returns:
        model with multiple outputs (for each layer in layer_names)
        -------
        """
        # Load pretrained VGG model
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          input_shape=INPUT_SHAPE,
                                          weights='imagenet')

        vgg.trainable = False

        outputs = [vgg.get_layer(layer[0]).output for layer in self.layer_names]
        model = tf.keras.Model(vgg.input, outputs)

        return model
