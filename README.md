# Neural Style Transfer
This repository contains my implementation of the [Neural Style Transfer paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). The code is based on the explanation given in Andrew Ng's [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/).

## What is NST?

This algorithm allows to transfer the style from one image (style image) to another (content image). The resulting (generated) image is initialized as a random noise sampled from the uniform/Gaussian distribution correlated with the content image. Rather than learning the parameters, the model is trained to update the pixel values of a generated image based on the loss between this image and the content and style images. This implementation uses a pre-trained 19-layer VGG network which is trained for 20000 epochs.

## Loss Function
### Content Cost
The content cost is computed as a squared difference between activation maps of the content image $C$ and the generated image $G$. The feature maps are usually taken from the layer in the middle of the network.

$$J_{content}(C, G) = \frac{1}{4 \times n_{H}^{\[l\]} \times n_{W}^{\[l]} \times n_{C}^{\[l\]}} \sum(a^{\[l\](C)} - a^{\[l\](G)})^2$$

### Style Cost

The style cost is defined as the *unnormalized cross covariance* between activation maps across channels.

The [Gram matrix](https://en.wikipedia.org/wiki/Gram_matrix) used to compute the style of a feature map at layer $l$ is given by:

$$ G_{(gram)kk'}^{\[l\](S)} = \sum_{i}^{n_{H}^{\[l\]}} \sum_{j}^{n_{W}^{\[l\]}} a_{i,j,k}^{\[l\](S)} a_{i,j,k'}^{\[l\](S)} $$
$$ G_{(gram)kk'}^{\[l\](G)} = \sum_{i}^{n_{H}^{\[l\]}} \sum_{j}^{n_{W}^{\[l\]}} a_{i,j,k}^{\[l\](G)} a_{i,j,k'}^{\[l\](G)} $$

for each $kk'$, where $a_{i,j,k}^{\[l\]}$ is the activation value of the feature map $k$ at point $i,j$.

This is is equal to:

$$ G_{(gram)}^{(A)} = AA^{T} $$ 

The style cost is computed as follows:

$$J_{style}(S, G) = \frac{1}{(4 \times n_{H}^{[l]} \times n_{W}^{[l]} \times n_{C}^{[l]})^2} \sum_{k} \sum_{k'} (G_{(gram)kk'}^{\[l\](S)} - G_{(gram)kk'}^{\[l\](G)})^2 $$

We can get better results if the style is computed from multiple layers and then combined. Each layer is weighted by some hyperparameter $\lambda$ which reflects how much the layer will contribute to the overall style:

$$J_{style} (S,G) = \sum_{l} \lambda^{[l]} J_{style}^{[l]} (S,G)$$

In this implementation all layers used for computing the style cost are assigned a value of $0.2$. Note that it is important that the sum of all layers is $1$.


### Total Cost

Total cost combines the content and style costs and weights them by additional hyperparameters $\alpha$ and $\beta$,

$$J_{total} (C, S, G) = \alpha J_{content} (C, G) + \beta J_{style} (S, G)$$

## Results
Here are some of the results of training the algorithm over 20,000 epochs.
![result_1](https://user-images.githubusercontent.com/49316611/202245901-86a635c4-4299-409c-912d-a7cdad7cb5bb.png)
![result_2](https://user-images.githubusercontent.com/49316611/202245909-c8166e7e-5bb9-4237-bd4d-d740994413a6.png)
![result_3](https://user-images.githubusercontent.com/49316611/202261524-72937fe7-7365-4616-9af1-27296dd992a6.png)
![result_4](https://user-images.githubusercontent.com/49316611/202463977-c9b8631a-fadf-44c0-9f48-15c826b301d8.png)

It can be seen that different values of alpha and beta slightly affect the final image:
<p align="center">
<img width="300" alt="ab1" src="https://user-images.githubusercontent.com/49316611/202463047-bce0bebf-7c0b-4fb2-8c16-9c6c7ba31f09.png">
<img width="300" alt="ab2" src="https://user-images.githubusercontent.com/49316611/202463060-5faa2538-d7e1-430b-97ae-dcd659564817.png">
<img width="300" alt="ab3" src="https://user-images.githubusercontent.com/49316611/202463068-9a9ab6a0-d68e-494b-8a0f-2812979e4d2a.png">
</p>

We can also see that using different types of noise leads to different results (left - uniform noise, right - gaussian noise):  

<p align="center">
<img width="247" alt="uniform" src="https://user-images.githubusercontent.com/49316611/202485847-c6501e35-b96a-4d9b-8eca-b23642c1f334.png">
<img width="250" alt="gaussian" src="https://user-images.githubusercontent.com/49316611/202485887-ec8f81a9-da3d-4578-8c74-72ee3c0a33ff.png">
</p>

This animation illustrates how the style transfer is performed starting from the original image+noise to the final artistic version.
<p align="center">
<img width="300" alt="gif" src="https://user-images.githubusercontent.com/49316611/202254073-813d856b-a34e-456f-b1cb-22e393e7b9d0.gif">
</p>
