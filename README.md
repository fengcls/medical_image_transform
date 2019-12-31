# Medical image transform
This repository includes a set of tools to transform (translate, synthesize, reconstruct) medical images.

The implementation is mostly based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Multiple generative methods are implemented, including mainly GAN based approach.

In GAN based approaches, WGAN+GP is used, multiple generators and discriminators are supported, including one model that uses 3D+2D discriminators.

The data as a running example is from a multi-modal MRI dataset BraTS (http://braintumorsegmentation.org/). And it provides inteface to easily work with any medical image dataset.

## Demo applications:

### cross-modality generation:

T1 <-> T2

T1 <-> T1 with contrast (pre_predict_post in this repo)

T1 with contrast <-> T2

...

