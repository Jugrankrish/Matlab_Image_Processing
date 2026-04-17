import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def create_sample_cover_image(path="cover.png", size=(512, 512)):
    w, h = size
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    low_freq = (
        0.4 * np.sin(2 * np.pi * 1.5 * xx) +
        0.3 * np.cos(2 * np.pi * 1.0 * yy) +
        0.2 * xx * yy
    )
    mid_freq = 0.15 * np.sin(2 * np.pi * 8 * xx) * np.cos(2 * np.pi * 6 * yy)
    rng = np.random.default_rng(42)
    high_freq = 0.08 * rng.standard_normal((h, w))

    combined = low_freq + mid_freq + high_freq
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    combined = (combined * 255).astype(np.uint8)

    img = Image.fromarray(combined, mode='L').convert('RGB')
    draw = ImageDraw.Draw(img)
    for py in range(h // 3):
        alpha = py / (h // 3)
        for px in range(w):
            r, g, b = img.getpixel((px, py))
            img.putpixel((px, py), (
                int(r * 0.6 + 30 * (1 - alpha)),
                int(g * 0.8 + 60 * (1 - alpha)),
                int(b * 1.0 + 80 * (1 - alpha))
            ))

    img.save(path)
    print(f"  Cover image saved → {path}")
    return path


def create_sample_watermark(path="watermark.png", size=(128, 128)):
    w, h = size
    img = Image.new('L', (w, h), color=0)
    draw = ImageDraw.Draw(img)

    margin = 10
    draw.ellipse([margin, margin, w - margin, h - margin], outline=255, width=6)

    cx, cy = w // 2, h // 2
    pts = [
        (cx - 35, cy - 25), (cx - 20, cy + 25), (cx, cy),
        (cx + 20, cy + 25), (cx + 35, cy - 25)
    ]
    draw.line(pts, fill=255, width=5)
    draw.ellipse([cx - 5, margin + 8, cx + 5, margin + 18], fill=255)

    img.save(path)
    print(f"  Watermark saved → {path}")
    return path


def load_cover_image(path):
    print(f"\n[STEP 1] Loading cover image: {path}")
    img = Image.open(path)
    img_gray = img.convert('L')
    gray_array = np.array(img_gray, dtype=np.float64)
    print(f"  Shape: {gray_array.shape}, Range: [{gray_array.min():.1f}, {gray_array.max():.1f}]")
    return gray_array, img


def load_watermark(path, target_size):
    print(f"\n[STEP 2] Loading watermark: {path}")
    wm_img = Image.open(path).convert('L')
    wm_resized = wm_img.resize(target_size, Image.LANCZOS)
    wm_array = np.array(wm_resized, dtype=np.float64)
    wm_binary = (wm_array > 128).astype(np.float64)
    print(f"  Resized to: {target_size}, White pixels: {int(wm_binary.sum())}")
    return wm_binary


def compute_fft(image_array):
    print("\n[STEP 3] Computing 2D FFT...")
    fft_result = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft_result)
    h, w = image_array.shape
    print(f"  FFT shape: {fft_shifted.shape}, dtype: {fft_shifted.dtype}")
    print(f"  DC magnitude: {np.abs(fft_shifted[h//2, w//2]):.1f}")
    return fft_shifted


def visualize_spectrum(fft_shifted):
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude


def embed_watermark(fft_shifted, watermark, strength=40, radius_percent=0.7):
    print("\n[STEP 4] Embedding watermark...")
    h, w = fft_shifted.shape
    cy, cx = h // 2, w // 2
    max_radius = np.sqrt(cx**2 + cy**2)
    inner_radius = radius_percent * max_radius * 0.5

    fft_watermarked = fft_shifted.copy()

    rows = np.arange(h) - cy
    cols = np.arange(w) - cx
    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
    distance_grid = np.sqrt(row_grid**2 + col_grid**2)

    freq_mask = (distance_grid >= inner_radius) & (distance_grid <= max_radius)
    embed_mask = freq_mask & (watermark > 0.3)

    print(f"  Embedding at {embed_mask.sum()} frequency locations (strength={strength})")

    embed_count = 0
    for r in range(h):
        for c in range(w):
            if embed_mask[r, c]:
                wm_val = watermark[r, c]
                fft_watermarked[r, c] += wm_val * strength
                mr = (h - r) % h
                mc = (w - c) % w
                fft_watermarked[mr, mc] += wm_val * strength
                embed_count += 1

    print(f"  Symmetric pairs: {embed_count} × 2 = {embed_count * 2} total modifications")
    return fft_watermarked


def reconstruct_image(fft_watermarked):
    print("\n[STEP 5] Running Inverse FFT...")
    fft_unshifted = np.fft.ifftshift(fft_watermarked)
    reconstructed = np.fft.ifft2(fft_unshifted)
    reconstructed_real = np.real(reconstructed)
    imaginary_residual = np.abs(np.imag(reconstructed)).max()
    print(f"  Max imaginary residual: {imaginary_residual:.2e}")
    reconstructed_clipped = np.clip(reconstructed_real, 0, 255)
    return reconstructed_clipped.astype(np.uint8)


def extract_watermark(watermarked_image_array, original_fft_shifted=None):
    print("\n[STEP 6] Extracting watermark via FFT...")
    fft_wm = np.fft.fft2(watermarked_image_array.astype(np.float64))
    fft_wm_shifted = np.fft.fftshift(fft_wm)
    mag_wm = np.log1p(np.abs(fft_wm_shifted))

    if original_fft_shifted is not None:
        mag_orig = np.abs(original_fft_shifted)
        mag_new = np.abs(fft_wm_shifted)
        diff = mag_new - mag_orig
        diff_positive = np.clip(diff, 0, None)
        print(f"  Max difference magnitude: {diff_positive.max():.2f}")
        return mag_wm, diff_positive

    return mag_wm, None


def compute_psnr(original, watermarked):
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(255.0**2 / mse)
    return psnr


def visualize_all_steps(original_gray, watermark, original_fft_spectrum,
                         watermarked_fft_spectrum, watermarked_image,
                         extracted_spectrum, diff_map, psnr_value):
    print("\n[STEP 7] Generating visualization...")

    fig = plt.figure(figsize=(20, 10), facecolor='#050810')
    fig.suptitle('FREQUENCY-DOMAIN IMAGE WATERMARKER — Full Pipeline Visualization',
                 fontsize=16, fontweight='bold', color='#00F5D4',
                 fontfamily='monospace', y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)

    def style_ax(ax, title, subtitle=""):
        ax.set_title(title, color='#00F5D4', fontsize=9, fontweight='bold',
                     fontfamily='monospace', pad=6)
        if subtitle:
            ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
                    ha='center', fontsize=7, color='#64748B', fontfamily='monospace')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#1E2D4A')
            spine.set_linewidth(1.5)
        ax.set_facecolor('#0C1020')

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_gray, cmap='gray', vmin=0, vmax=255)
    style_ax(ax1, "01 · ORIGINAL IMAGE", "Cover photograph (grayscale)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(watermark, cmap='gray', vmin=0, vmax=1)
    style_ax(ax2, "02 · WATERMARK LOGO", "Secret binary symbol (B&W)")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(original_fft_spectrum, cmap='plasma')
    style_ax(ax3, "03 · ORIGINAL FFT SPECTRUM", "log(|FFT|) — DC at center")
    ax3.text(original_fft_spectrum.shape[1] // 2, original_fft_spectrum.shape[0] // 2,
             '+', color='white', ha='center', va='center', fontsize=16, fontweight='bold')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(watermarked_fft_spectrum, cmap='plasma')
    style_ax(ax4, "04 · WATERMARKED FFT SPECTRUM", "Watermark glows at outer edges")

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(watermarked_image, cmap='gray', vmin=0, vmax=255)
    style_ax(ax5, "05 · WATERMARKED IMAGE", f"Visually identical · PSNR={psnr_value:.1f}dB")
    ax5.text(0.02, 0.98, f"PSNR: {psnr_value:.1f} dB", transform=ax5.transAxes,
             va='top', fontsize=8, color='#4ADE80', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0C1020', alpha=0.8))

    ax6 = fig.add_subplot(gs[1, 1])
    if diff_map is not None:
        ax6.imshow(np.log1p(diff_map), cmap='hot')
        style_ax(ax6, "06 · EXTRACTED WATERMARK", "FFT difference reveals the mark")
    else:
        ax6.imshow(watermarked_fft_spectrum, cmap='hot')
        style_ax(ax6, "06 · WATERMARKED SPECTRUM", "Watermark visible in high-freq ring")

    ax7 = fig.add_subplot(gs[1, 2])
    pixel_diff = np.abs(original_gray.astype(np.float64) - watermarked_image.astype(np.float64))
    h, w = pixel_diff.shape
    cy, cx = h // 2, w // 2
    zoom = pixel_diff[cy - 100:cy + 100, cx - 100:cx + 100]
    im7 = ax7.imshow(zoom * 10, cmap='inferno', vmin=0, vmax=30)
    style_ax(ax7, "07 · PIXEL DIFFERENCE (×10)", "Microscopic changes in texture")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.set_facecolor('#0C1020')
    style_ax(ax8, "08 · PIPELINE STATS", "")

    stats = [
        ("IMAGE SIZE", f"{original_gray.shape[1]}×{original_gray.shape[0]} px"),
        ("PSNR", f"{psnr_value:.2f} dB"),
        ("QUALITY", "EXCELLENT" if psnr_value > 40 else "GOOD" if psnr_value > 35 else "FAIR"),
        ("MAX PIX DIFF", f"{pixel_diff.max():.1f} / 255"),
        ("MEAN PIX DIFF", f"{pixel_diff.mean():.3f}"),
        ("WATERMARK PX", f"{int(watermark.sum())} white pixels"),
        ("ALGORITHM", "Cooley-Tukey"),
        ("SYMMETRY", "Hermitian ✓"),
    ]

    y_pos = 0.92
    for label, value in stats:
        color = '#4ADE80' if label == "QUALITY" and psnr_value > 40 else '#00F5D4'
        ax8.text(0.05, y_pos, f"{label}:", transform=ax8.transAxes,
                 fontsize=8, color='#64748B', fontfamily='monospace', va='top')
        ax8.text(0.55, y_pos, value, transform=ax8.transAxes,
                 fontsize=8, color=color, fontfamily='monospace', va='top', fontweight='bold')
        y_pos -= 0.115

    output_path = "watermarker_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#050810', edgecolor='none')
    print(f"  Visualization saved → {output_path}")
    plt.close()
    return output_path


def main(cover_path=None, watermark_path=None, strength=40, radius_percent=0.7):
    print("=" * 60)
    print("  FREQUENCY-DOMAIN IMAGE WATERMARKER")
    print("=" * 60)

    os.makedirs("temp", exist_ok=True)

    if cover_path is None or not os.path.exists(cover_path):
        cover_path = "temp/cover.png"
        create_sample_cover_image(cover_path, size=(512, 512))

    if watermark_path is None or not os.path.exists(watermark_path):
        watermark_path = "temp/watermark.png"
        create_sample_watermark(watermark_path, size=(128, 128))

    original_gray, original_pil = load_cover_image(cover_path)
    h, w = original_gray.shape

    watermark = load_watermark(watermark_path, target_size=(w, h))

    fft_shifted = compute_fft(original_gray)
    original_fft_spectrum = visualize_spectrum(fft_shifted)

    fft_watermarked = embed_watermark(fft_shifted, watermark, strength=strength,
                                       radius_percent=radius_percent)

    watermarked_image = reconstruct_image(fft_watermarked)

    fft_wm_shifted = np.fft.fftshift(np.fft.fft2(watermarked_image.astype(np.float64)))
    watermarked_fft_spectrum = visualize_spectrum(fft_wm_shifted)

    extracted_spectrum, diff_map = extract_watermark(watermarked_image, fft_shifted)

    print("\n[STEP 7] Computing quality metrics...")
    psnr = compute_psnr(original_gray, watermarked_image)
    pixel_diff = np.abs(original_gray - watermarked_image.astype(np.float64))
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Max pixel diff: {pixel_diff.max():.1f} / 255")
    print(f"  Mean pixel diff: {pixel_diff.mean():.3f} / 255")

    Image.fromarray(watermarked_image, mode='L').save("watermarked_image.png")
    print("\n  Watermarked image saved → watermarked_image.png")

    visualize_all_steps(original_gray, watermark, original_fft_spectrum,
                        watermarked_fft_spectrum, watermarked_image,
                        extracted_spectrum, diff_map, psnr)

    print("\n" + "=" * 60)
    print("  DONE!")
    print(f"  PSNR: {psnr:.2f} dB — watermark is invisible")
    print("=" * 60)


if __name__ == "__main__":
    main(
        cover_path="C:\Users\ACER\OneDrive\Pictures\Screenshots\Screenshot 2026-03-28 150044.png",
        watermark_path="C:\Users\ACER\Downloads\Untitled.jpg",
        strength=40,
        radius_percent=0.7
    )