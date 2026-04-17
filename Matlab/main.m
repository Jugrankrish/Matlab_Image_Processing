function fft_watermarker()
    cover  = double(imresize(rgb2gray(imread('D:\SignalProcessingProject\Matlab\cover.jpeg')),  [512 512]));
    wm_raw = double(imresize(rgb2gray(imread('D:\SignalProcessingProject\Matlab\watermark.jpeg')), [128 128]));
    wm_bin = edge(im2double(uint8(wm_raw > 128) * 255), 'canny');
    strength = 10000;

    F_orig = fftshift(fft2(cover));
    F = F_orig;
    [H, W] = size(F);
    r0 = floor(H/2 - 64);
    c0 = floor(W/2 - 64);

    for i = 1:128
        for j = 1:128
            if wm_bin(i,j)
                r = r0+i; c = c0+j;
                F(r,c) = F(r,c) + strength;
                F(mod(H-r,H)+1, mod(W-c,W)+1) = F(mod(H-r,H)+1, mod(W-c,W)+1) + strength;
            end
        end
    end

    rec = real(ifft2(ifftshift(F)));
    rec = rec - min(rec(:));
    watermarked = uint8(rec / max(rec(:)) * 255);
    
    diff_mag = abs(F) - abs(F_orig);
    diff_mag = max(diff_mag, 0);
    cx = H/2; cy = W/2; half = 128;
    patch = diff_mag(cx-half+1:cx+half, cy-half+1:cy+half);
    revealed = patch / max(patch(:));

    mse = mean((cover(:) - double(watermarked(:))).^2);
    psnr_val = 10 * log10(255^2 / mse);
    fprintf('PSNR: %.2f dB\n', psnr_val);

    imwrite(watermarked,           'watermarked.png');
    imwrite(uint8(revealed * 255), 'watermark_revealed.png');

    figure('Position', [100 100 1400 400]);
    subplot(1,4,1); imshow(uint8(cover));
    title('1. Original image'); xlabel('Clean — no watermark');

    subplot(1,4,2); imshow(uint8(wm_bin) * 255);
    title('2. Watermark logo'); xlabel('Secret binary pattern');

    subplot(1,4,3); imshow(watermarked);
    title(sprintf('3. Watermarked (PSNR = %.1f dB)', psnr_val));
    xlabel('Looks identical to original');

    subplot(1,4,4); imagesc(revealed, [0 1]); colormap(gca, gray); axis image off;
    title('4. Revealed via FFT magnitude'); xlabel('S glows in frequency domain');
end