# Neural Style Transfer
This repository contains my implementation of the [Neural Style Transfer paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

Similarly to the original paper it uses a 19-layer VGG network.

![](https://github.com/iamkzntsv/neural-style-transfer/blob/main/nst.gif)

## Loss Function
### Content Cost
The content cost is computed as a squared difference between activation maps of the content image $C$ and the generated image $G$. The feature maps are usually taken from the layer in the middle of the network.

$$\mathbb{L}_{content}(C, G) = \frac{1}{4n_{H}^{[l]} \times n_{W}^{[l]} \times n_{C}^{[l]}} (a^{[l]} - a^{[l]})^2$$

### Style Cost

The style cost is defined as the *unnormalized cross covariance* between activation maps across channels.
First we compute *Gram matrix*:

Let $a_{i,j,k}^[l] = $ activation at $(i,j,k)$

$$\mathbb{L}_{content}(S, G) = \frac{1}{4n_{H}^{[l]}n_{W}^{[l]}n_{C}^{[l]}}$$

### Total Cost

$$\mathbb{L}_{total}(C, G) = \alpha \mathbb{L}_{content}(C, G) + \beta \mathbb{L}_{style}(S, G)$$