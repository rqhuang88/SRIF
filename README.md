# SRIF
Official implementation of SIGGRAPH Asia paper -- SRIF: Semantic Shape Registration Empowered by Diffusion-based Image Morphing and Flow Estimation

# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```
# Datasets
## Image Morph
Given two shapes, we first render multi-view images by:
```bash
python render_folder.py
```
Then use DiffMorpher [1] to generate image interpolation with respect to each views.

## Dynamic 3D Gaussian
We convert the image interpolation results to SC-GS [2] inputs by:
```bash
python conver_npy_to_dataset.py
```
Then use SC-GS [2] to obtain a set of dense and noisy point clouds.

# Training
```bash
python train.py
```

# Testing
```bash
python test.py
```

# References
[1]: Zhang, K., Zhou, Y., Xu, X., Dai, B., & Pan, X. (2024). DiffMorpher: Unleashing the Capability of Diffusion Models for Image Morphing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

[2]: Huang, Y. H., Sun, Y. T., Yang, Z., Lyu, X., Cao, Y. P., & Qi, X. (2024). Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4220-4230).
