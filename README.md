This project presents a **conditional image-to-image diffusion model** designed to transform sunrise images into their corresponding sunset variants. It is implemented in **PyTorch** and built upon a custom **U-Net architecture**, integrating time embeddings and skip connections for robust denoising. The model is trained on paired image data and leverages the **Denoising Diffusion Probabilistic Model (DDPM)** framework to learn the reverse process of image degradation.

## 🌐 Project Description

Diffusion models have emerged as a powerful family of generative models that iteratively refine noisy images to generate realistic samples. This implementation focuses on **conditional generation**, where the model uses a guiding input image (sunrise) to predict a desired target image (sunset).

The process involves gradually adding noise to a target image over a series of time steps and then training the model to predict and remove that noise, conditioned on the original guiding image. Over time, the model learns the reverse diffusion process—essentially, how to go from pure noise back to the original image, guided by context.

## 🧠 Technical Details

### Model Architecture
- **Custom U-Net Backbone**: Encoder-decoder with skip connections.
- **Conditioning**: Input image and guiding condition are concatenated along the channel dimension.
- **Time Embeddings**: Injected into the bottleneck via MLPs and projected into spatial dimensions.
- **Normalization**: BatchNorm used throughout.
- **Activation**: ReLU in encoder/decoder and SiLU for time embedding.

### Diffusion Process
- **Timesteps**: 1000 steps using cosine beta schedule for smoother transitions.
- **Noise Schedule**: Cosine schedule improves stability over linear betas.
- **Forward Process (q_sample)**: Adds Gaussian noise to images at each timestep.
- **Reverse Process (denoising)**: Model predicts the added noise at each timestep.

### Loss Function
- Combines **MSE (L2)** and **MAE (L1)** losses for better convergence and finer details.
- Loss = `MSE(pred, noise) + 0.1 * MAE(pred, noise)`

### Training Strategy
- **Data Augmentation**: Random rotations, perspective transforms, and color jitter applied equally to input-output pairs.
- **Optimizer**: AdamW with `lr=1e-4`
- **Scheduler**: StepLR reduces LR by 0.5 every 100 epochs.
- **Gradient Clipping**: Max norm of 1.0 to stabilize training.
- **Epochs**: Default 50 (configurable).
- **Checkpointing**: Models saved periodically.

## 🔍 Dataset

The dataset consists of **paired sunrise and sunset images**. During training, both are transformed and augmented consistently. A custom `ImagePairDataset` class is used to handle loading and preprocessing.

## 🏁 Generation Process

To generate a sunset-style image:
1. Begin with random noise.
2. Iterate through reverse timesteps.
3. At each step:
   - Predict noise using the model.
   - Refine the image using the predicted noise.
   - Optionally save intermediate images.

The final output is rescaled to `[0, 1]` and saved for visualization.

## 📂 Output Artifacts
- `diffusion_outputs/`: Intermediate and final images from the generation process.
- `training_samples/`: Model outputs after each epoch.
- `training_loss.png`: Loss graph for training performance.
- `model_checkpoints/`: Checkpointed `.pth` model files.

## 🔧 Requirements
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- PIL
- tqdm

## ✨ Potential Improvements
- Use **attention mechanisms** (e.g., self/cross-attention).
- Experiment with **DDIM** or **Latent Diffusion**.
- Switch to **Patch-based Conditioning** for localized translation.
- Add **CLIP-based losses** for semantic alignment.

## 📌 Use Case
This model is suitable for tasks like:
- Artistic image translation (e.g., style or time-of-day transfer)
- Data augmentation for vision models
- Educational use in understanding generative diffusion models

## 🔍 Example Usage

```bash
python diffusionmodel.py
```

The script trains the model and periodically generates sample outputs.

> Built with ❤️ using PyTorch. Contributions and forks welcome!
