
# how to install (diffusion only)
conda create -n diffusion python=3.9
conda activate diffusion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets diffusers[torch]
pip install omegaconf
pip install kornia
pip install hydra-core
pip install tensorboard
pip install pytorch-lightning
pip install git+https://github.com/openai/CLIP.git

# how to install (+ differential rendering with uv unwrapping)
pip install pytorch_msssim
pip install trimesh
pip install rembg
pip install meshzoo
pip install open3d
pip install smplx
pip install ninja
pip install tbb
pip install UVTextureConverter
cd nvdiffrast & pip install .
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# in case of cuda conflict (force reinstall) - depending on which version of cuda is installed in your pc.
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
