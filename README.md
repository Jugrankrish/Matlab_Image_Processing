# Invisible Image Watermarking via FFT

This project demonstrates a digital signal processing (DSP) technique for embedding an invisible, secret watermark into an image. Instead of altering the pixels directly in the spatial domain, this script uses the 2-Dimensional Fast Fourier Transform (2D-FFT) to hide the watermark within the frequency domain of the cover image.

## 🚀 Features
* **Frequency Domain Embedding:** Modifies the magnitude of specific frequency components to hide data, making the watermark invisible to the naked eye.
* **Conjugate Symmetry Preservation:** Ensures the modified frequency spectrum remains symmetric so the inverse FFT produces a valid, real-world image without complex artifacts.
* **Edge-Based Watermarking:** Uses Canny edge detection on the secret logo to extract its most defining features before embedding, optimizing the payload size.
* **Quality Metrics:** Automatically calculates the Peak Signal-to-Noise Ratio (PSNR) to quantify how identical the watermarked image is to the original.

## 🧠 How It Works
1. **Preprocessing:** The cover image is resized to 512x512 and converted to grayscale. The secret watermark is resized to 128x128, converted to binary, and passed through a Canny edge detector.
2. **Transform:** The script computes the 2D-FFT of the cover image and shifts the zero-frequency components to the center.
3. **Embedding:** The binary edges of the watermark are added to the mid-frequency bands of the cover image's FFT. The modifications are mirrored symmetrically to ensure the final image remains real (not complex).
4. **Reconstruction:** An Inverse FFT (IFFT) is applied to return the image to the spatial domain. 
5. **Extraction:** To reveal the watermark, the script subtracts the magnitude of the original image's FFT from the watermarked image's FFT, isolating the hidden pattern.

## 🛠️ Getting Started

### Prerequisites
* MATLAB installed on your machine.
* **Image Processing Toolbox** (required for functions like `edge()`, `rgb2gray()`, and `fft2()`).

### Installation & Setup
1. Clone this repository to your local machine.
2. Place a cover image (`cover.jpeg`) and a watermark logo (`watermark.jpeg`) in the project directory.
3. **Important:** Update the file paths in `fft_watermarker.m` to point to your local images. 
   *(e.g., change `D:\SignalProcessingProject\...` to your actual directory, or use relative paths).*

### Running the Code
Simply execute the `fft_watermarker()` function in your MATLAB command window. 

## 📊 Outputs
Running the script generates the following outputs:
* `watermarked.png`: The final image containing the hidden data.
* `watermark_revealed.png`: The extracted frequency-domain representation showing the hidden logo.
* **Comparison Figure:** A 4-panel MATLAB plot showing:
  1. The clean original image.
  2. The secret binary edge pattern.
  3. The watermarked image (with calculated PSNR).
  4. The glowing watermark revealed via FFT magnitude.
