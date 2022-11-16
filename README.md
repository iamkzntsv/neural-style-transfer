# Neural Style Transfer
This repository contains my implementation of the [Neural Style Transfer paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

Similarly to the original paper it uses a 19-layer VGG network.

![](https://github.com/iamkzntsv/neural-style-transfer/blob/main/nst.gif)

## Loss Function
### Content Cost
The content cost is computed as a squared difference between activation maps of the content image $C$ and the generated image $G$. The feature maps are usually taken from the layer in the middle of the network.

$$\mathcal{L}_{content}(C, G) = \frac{1}{4 \times n_{H}^{\[l\]} \times n_{W}^{\[l]} \times n_{C}^{\[l\]}} \sum(a^{\[l\]} - a^{\[l\]})^2$$

### Style Cost

The style cost is defined as the *unnormalized cross covariance* between activation maps across channels.
First we compute the *Gram matrix*:

The activation value of the feature map $k$ at point $i,j$ is defined as:

$$a_{i,j,k}^{\[l\]}$$

The Gram matrix for a single layer of content and generated images is given as follows:

$$ G_{(gram)kk'}^{\[l\](S)} = \sum_{i}^{n_{H}^{\[l\]}} \sum_{j}^{n_{W}^{\[l\]}} a_{i,j,k}^{\[l\](S)} a_{i,j,k'}^{\[l\](S)} $$
$$ G_{(gram)kk'}^{\[l\](G)} = \sum_{i}^{n_{H}^{\[l\]}} \sum_{j}^{n_{W}^{\[l\]}} a_{i,j,k}^{\[l\](G)} a_{i,j,k'}^{\[l\](G)} $$

for each $kk'$

More generally this is equal to:

$$ G_{(gram)}^(A) = AA^{T}

The style cost is computed as follows:

$$\mathcal{L}_{style}(S, G) = \frac{1}{4n_{H}^{[l]}n_{W}^{[l]}n_{C}^{[l]}} \sum_{k} \sum_{k'} (G_{(gram)kk'}^{\[l\](S)} - G_{(gram)kk'}^{\[l\](G)})^2 $$

### Total Cost

$$\mathcal{L}_{total}(C, G) = \alpha \mathcal{L}_{content}(C, G) + \beta \mathcal{L}_{style}(S, G)$$