%% 实验七：图像锐化实验
clear; close all; clc;
fprintf('=== 图像锐化实验 ===\n\n');

%% 1. 实验原理说明
fprintf('实验原理：\n');
fprintf('图像锐化的目的是加强图像中景物的细节边缘和轮廓。\n');
fprintf('锐化的作用是使灰度反差增强，因为边缘和轮廓都位于灰度突变的地方。\n');
fprintf('锐化算法基于微分作用，主要包括梯度法、Sobel算子法、Laplace算子法和高频加重滤波法。\n\n');

%% 2. 加载原始图像
fprintf('加载原始图像...\n');
original_img = imread("D:\Picture\20250131\B0D0137B0CE3E8810718BB828CBD5D6C.jpg");

% 确保图像是灰度图像
if size(original_img, 3) == 3
    original_img = rgb2gray(original_img);
end

% 调整图像大小以加快处理速度
original_img = imresize(original_img, [256, 256]);

% 显示图像信息
[m, n] = size(original_img);
fprintf('图像尺寸: %d × %d 像素\n', m, n);

%% 3. 显示原始图像
figure('Name', '原始图像与噪声图像', 'Position', [100, 100, 1200, 600]);

% 原始图像
subplot(1, 2, 1);
imshow(original_img);
title('原始图像');

% 原始图像的3D表面图
subplot(1, 2, 2);
[X, Y] = meshgrid(1:4:n, 1:4:m);
Z = double(original_img(1:4:end, 1:4:end));
surf(X, Y, Z, 'EdgeColor', 'none');
title('原始图像3D表面图');
xlabel('X轴');
ylabel('Y轴');
zlabel('灰度值');
colormap(gray);
view(45, 30);

%% 4. 添加噪声并降噪处理（验证先降噪后锐化的必要性）
fprintf('\n添加噪声并降噪处理...\n');

% 添加高斯噪声
noisy_img = imnoise(original_img, 'gaussian', 0, 0.01);

% 添加椒盐噪声
salt_pepper_img = imnoise(original_img, 'salt & pepper', 0.05);

% 对噪声图像进行降噪处理（高斯滤波）
sigma = 1;
kernel_size = 5;
gaussian_kernel = fspecial('gaussian', kernel_size, sigma);
denoised_gaussian = imfilter(noisy_img, gaussian_kernel, 'replicate');
denoised_salt_pepper = medfilt2(salt_pepper_img, [3 3]);

% 显示噪声和降噪效果
figure('Name', '噪声与降噪效果', 'Position', [150, 150, 1200, 800]);

% 高斯噪声图像
subplot(3, 4, 1);
imshow(noisy_img);
title('高斯噪声图像');

subplot(3, 4, 2);
imshow(denoised_gaussian);
title('高斯滤波降噪后');

% 椒盐噪声图像
subplot(3, 4, 3);
imshow(salt_pepper_img);
title('椒盐噪声图像');

subplot(3, 4, 4);
imshow(denoised_salt_pepper);
title('中值滤波降噪后');

% 计算信噪比
snr_noisy = 20 * log10(norm(double(original_img(:))) / norm(double(noisy_img(:)) - double(original_img(:))));
snr_denoised = 20 * log10(norm(double(original_img(:))) / norm(double(denoised_gaussian(:)) - double(original_img(:))));

fprintf('信噪比对比:\n');
fprintf('  噪声图像SNR: %.2f dB\n', snr_noisy);
fprintf('  降噪后图像SNR: %.2f dB\n', snr_denoised);

%% 5. 梯度法锐化（Roberts算子）
fprintf('\n使用Roberts算子进行锐化...\n');

% 对原始图像使用Roberts算子
img_roberts = roberts_sharpen(original_img);

% 对噪声图像直接使用Roberts算子（不降噪）
noisy_roberts = roberts_sharpen(noisy_img);

% 对降噪后图像使用Roberts算子
denoised_roberts = roberts_sharpen(denoised_gaussian);

% 显示Roberts算子锐化结果
figure('Name', 'Roberts算子锐化效果', 'Position', [100, 100, 1200, 800]);

% 原始图像锐化
subplot(3, 4, 1);
imshow(original_img);
title('原始图像');

subplot(3, 4, 2);
imshow(img_roberts);
title('Roberts算子锐化');

% 噪声图像锐化（不降噪）
subplot(3, 4, 5);
imshow(noisy_img);
title('高斯噪声图像');

subplot(3, 4, 6);
imshow(noisy_roberts);
title('直接锐化（噪声放大）');

% 降噪后锐化
subplot(3, 4, 9);
imshow(denoised_gaussian);
title('降噪后图像');

subplot(3, 4, 10);
imshow(denoised_roberts);
title('降噪后锐化');

% 计算边缘增强程度
edge_enhancement_original = compute_edge_enhancement(original_img, img_roberts);
edge_enhancement_noisy = compute_edge_enhancement(noisy_img, noisy_roberts);
edge_enhancement_denoised = compute_edge_enhancement(denoised_gaussian, denoised_roberts);

subplot(3, 4, [3, 4, 7, 8, 11, 12]);
bar([1, 2, 3], [edge_enhancement_original, edge_enhancement_noisy, edge_enhancement_denoised], ...
    'FaceColor', [0.6 0.6 0.8]);
set(gca, 'XTickLabel', {'原始图像锐化', '噪声图像直接锐化', '降噪后锐化'});
ylabel('边缘增强程度 (%)');
title('Roberts算子锐化效果对比');
grid on;

% 添加数值标签
text(1, edge_enhancement_original, sprintf('%.2f%%', edge_enhancement_original), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(2, edge_enhancement_noisy, sprintf('%.2f%%', edge_enhancement_noisy), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(3, edge_enhancement_denoised, sprintf('%.2f%%', edge_enhancement_denoised), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);

%% 6. Sobel算子锐化
fprintf('\n使用Sobel算子进行锐化...\n');

% 对原始图像使用Sobel算子
img_sobel = sobel_sharpen(original_img);

% 对噪声图像直接使用Sobel算子（不降噪）
noisy_sobel = sobel_sharpen(noisy_img);

% 对降噪后图像使用Sobel算子
denoised_sobel = sobel_sharpen(denoised_gaussian);

% 显示Sobel算子锐化结果
figure('Name', 'Sobel算子锐化效果', 'Position', [150, 150, 1200, 800]);

% 原始图像锐化
subplot(2, 3, 1);
imshow(original_img);
title('原始图像');

subplot(2, 3, 2);
imshow(img_sobel);
title('Sobel算子锐化');

% 梯度幅值显示
subplot(2, 3, 3);
[grad_mag, grad_dir] = sobel_gradient(original_img);
imshow(grad_mag, []);
title('Sobel梯度幅值');
colorbar;

% 噪声图像锐化（不降噪）
subplot(2, 3, 4);
imshow(noisy_sobel);
title('噪声图像直接锐化');

% 降噪后锐化
subplot(2, 3, 5);
imshow(denoised_sobel);
title('降噪后锐化');

% 梯度方向显示
subplot(2, 3, 6);
imshow(grad_dir, []);
title('Sobel梯度方向');
colorbar;

%% 7. Laplace算子锐化
fprintf('\n使用Laplace算子进行锐化...\n');

% 不同形式的Laplace算子
laplace_kernel_4 = [0 1 0; 1 -4 1; 0 1 0];  % 4邻域
laplace_kernel_8 = [1 1 1; 1 -8 1; 1 1 1];   % 8邻域

% 对原始图像使用不同Laplace算子
img_laplace_4 = laplace_sharpen(original_img, laplace_kernel_4);
img_laplace_8 = laplace_sharpen(original_img, laplace_kernel_8);

% 对降噪后图像使用Laplace算子
denoised_laplace_4 = laplace_sharpen(denoised_gaussian, laplace_kernel_4);

% 显示Laplace算子锐化结果
figure('Name', 'Laplace算子锐化效果', 'Position', [100, 100, 1200, 800]);

% 原始图像和4邻域Laplace锐化
subplot(3, 4, 1);
imshow(original_img);
title('原始图像');

subplot(3, 4, 2);
imshow(img_laplace_4);
title('4邻域Laplace锐化');

% 8邻域Laplace锐化
subplot(3, 4, 3);
imshow(img_laplace_8);
title('8邻域Laplace锐化');

% 降噪后Laplace锐化
subplot(3, 4, 4);
imshow(denoised_laplace_4);
title('降噪后4邻域Laplace锐化');

% 显示Laplace算子响应
subplot(3, 4, 5);
laplace_response_4 = conv2(double(original_img), laplace_kernel_4, 'same');
imshow(laplace_response_4, []);
title('4邻域Laplace响应');
colorbar;

subplot(3, 4, 6);
laplace_response_8 = conv2(double(original_img), laplace_kernel_8, 'same');
imshow(laplace_response_8, []);
title('8邻域Laplace响应');
colorbar;

% 频率响应分析
subplot(3, 4, [7, 8]);
freq_response_4 = abs(fftshift(fft2(laplace_kernel_4, 64, 64)));
freq_response_8 = abs(fftshift(fft2(laplace_kernel_8, 64, 64)));

surf(1:64, 1:64, log(freq_response_4 + 1), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
surf(1:64, 1:64, log(freq_response_8 + 1), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
title('Laplace算子频率响应（对数显示）');
xlabel('水平频率');
ylabel('垂直频率');
zlabel('幅度（对数）');
colormap(jet);
view(45, 30);
legend('4邻域', '8邻域');

% 计算锐化效果指标
sharpness_4 = compute_sharpness(img_laplace_4);
sharpness_8 = compute_sharpness(img_laplace_8);
sharpness_denoised = compute_sharpness(denoised_laplace_4);

subplot(3, 4, [9, 10, 11, 12]);
bar([1, 2, 3], [sharpness_4, sharpness_8, sharpness_denoised], ...
    'FaceColor', [0.8 0.6 0.6]);
set(gca, 'XTickLabel', {'4邻域Laplace', '8邻域Laplace', '降噪后Laplace'});
ylabel('锐化度指标');
title('Laplace算子锐化效果对比');
grid on;

% 添加数值标签
text(1, sharpness_4, sprintf('%.4f', sharpness_4), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(2, sharpness_8, sprintf('%.4f', sharpness_8), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(3, sharpness_denoised, sprintf('%.4f', sharpness_denoised), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);

%% 8. 高频加重滤波法（非锐化掩模）
fprintf('\n使用高频加重滤波法（非锐化掩模）进行锐化...\n');

% 不同增强系数
k_values = [0.5, 1.0, 1.5, 2.0];

% 对原始图像使用不同k值的非锐化掩模
unsharp_results = cell(length(k_values), 1);
for i = 1:length(k_values)
    unsharp_results{i} = unsharp_mask(original_img, k_values(i));
end

% 对降噪后图像使用非锐化掩模
denoised_unsharp = unsharp_mask(denoised_gaussian, 1.0);

% 显示非锐化掩模锐化结果
figure('Name', '非锐化掩模锐化效果', 'Position', [150, 150, 1200, 800]);

% 原始图像
subplot(3, 5, 1);
imshow(original_img);
title('原始图像');

% 不同k值的锐化结果
for i = 1:length(k_values)
    subplot(3, 5, i+1);
    imshow(unsharp_results{i});
    title(sprintf('非锐化掩模 k=%.1f', k_values(i)));
end

% 降噪后锐化
subplot(3, 5, 6);
imshow(denoised_unsharp);
title('降噪后非锐化掩模 (k=1.0)');

% 显示细节对比（局部放大）
subplot(3, 5, [7, 8, 9, 10]);
% 选择图像的一个区域进行局部放大
region = original_img(100:150, 100:150);
region_unsharp = unsharp_results{2}(100:150, 100:150); % k=1.0

imshowpair(region, region_unsharp, 'montage');
title('局部细节对比（左：原始，右：锐化）');

% 计算不同k值的锐化度
sharpness_values = zeros(length(k_values), 1);
for i = 1:length(k_values)
    sharpness_values(i) = compute_sharpness(unsharp_results{i});
end

subplot(3, 5, [11, 12, 13, 14, 15]);
plot(k_values, sharpness_values, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('增强系数 k');
ylabel('锐化度指标');
title('锐化度随增强系数的变化');
grid on;

%% 9. 综合对比：不同锐化方法的效果
fprintf('\n综合对比不同锐化方法的效果...\n');

% 对降噪后图像应用各种锐化方法
test_img = denoised_gaussian;

% 各种锐化方法
roberts_sharp = roberts_sharpen(test_img);
sobel_sharp = sobel_sharpen(test_img);
laplace_sharp_4 = laplace_sharpen(test_img, laplace_kernel_4);
unsharp_sharp = unsharp_mask(test_img, 1.0);

% 显示综合对比结果
figure('Name', '不同锐化方法综合对比', 'Position', [100, 100, 1400, 1000]);

% 原始降噪图像
subplot(3, 5, 1);
imshow(test_img);
title('降噪后原始图像');

% Roberts算子锐化
subplot(3, 5, 2);
imshow(roberts_sharp);
title('Roberts算子锐化');

% Sobel算子锐化
subplot(3, 5, 3);
imshow(sobel_sharp);
title('Sobel算子锐化');

% Laplace算子锐化（4邻域）
subplot(3, 5, 4);
imshow(laplace_sharp_4);
title('Laplace算子锐化（4邻域）');

% 非锐化掩模
subplot(3, 5, 5);
imshow(unsharp_sharp);
title('非锐化掩模 (k=1.0)');

% 计算各种方法的锐化度
sharpness_roberts = compute_sharpness(roberts_sharp);
sharpness_sobel = compute_sharpness(sobel_sharp);
sharpness_laplace = compute_sharpness(laplace_sharp_4);
sharpness_unsharp = compute_sharpness(unsharp_sharp);

% 显示锐化度对比
subplot(3, 5, [6, 7, 8]);
bar(1:4, [sharpness_roberts, sharpness_sobel, sharpness_laplace, sharpness_unsharp], ...
    'FaceColor', [0.6 0.8 0.6]);
set(gca, 'XTickLabel', {'Roberts', 'Sobel', 'Laplace', '非锐化掩模'});
ylabel('锐化度指标');
title('不同锐化方法的锐化度对比');
grid on;

% 添加数值标签
text(1, sharpness_roberts, sprintf('%.4f', sharpness_roberts), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(2, sharpness_sobel, sprintf('%.4f', sharpness_sobel), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(3, sharpness_laplace, sprintf('%.4f', sharpness_laplace), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(4, sharpness_unsharp, sprintf('%.4f', sharpness_unsharp), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);

% 计算边缘保持指标
edge_preservation_roberts = compute_edge_preservation(test_img, roberts_sharp);
edge_preservation_sobel = compute_edge_preservation(test_img, sobel_sharp);
edge_preservation_laplace = compute_edge_preservation(test_img, laplace_sharp_4);
edge_preservation_unsharp = compute_edge_preservation(test_img, unsharp_sharp);

% 显示边缘保持对比
subplot(3, 5, [9, 10]);
bar(1:4, [edge_preservation_roberts, edge_preservation_sobel, ...
          edge_preservation_laplace, edge_preservation_unsharp], ...
    'FaceColor', [0.8 0.6 0.8]);
set(gca, 'XTickLabel', {'Roberts', 'Sobel', 'Laplace', '非锐化掩模'});
ylabel('边缘保持指标 (%)');
title('不同锐化方法的边缘保持能力');
ylim([0 100]);
grid on;

% 显示局部细节对比
subplot(3, 5, [11, 12, 13, 14, 15]);
% 选择图像的一个区域进行局部放大
detail_region = test_img(120:180, 120:180);
detail_roberts = roberts_sharp(120:180, 120:180);
detail_sobel = sobel_sharp(120:180, 120:180);
detail_laplace = laplace_sharp_4(120:180, 120:180);
detail_unsharp = unsharp_sharp(120:180, 120:180);

% 创建蒙太奇图像
montage_img = [detail_region, detail_roberts, detail_sobel, detail_laplace, detail_unsharp];
montage_img = imresize(montage_img, [size(montage_img,1)*2, size(montage_img,2)*2]);
imshow(montage_img);
title('局部细节对比（从左到右：原始、Roberts、Sobel、Laplace、非锐化掩模）');

%% 10. 图像锐化的实际应用：文本增强
fprintf('\n图像锐化的实际应用：文本增强...\n');

% 创建一个包含文本的测试图像
text_img = uint8(200 * ones(256, 256));
text_img = insertText(text_img, [50, 100], 'Image', 'FontSize', 40, 'TextColor', 'black', 'BoxOpacity', 0);
text_img = insertText(text_img, [50, 150], 'Sharpening', 'FontSize', 40, 'TextColor', 'black', 'BoxOpacity', 0);
text_img = rgb2gray(text_img); % 转换为灰度图像

% 添加轻微模糊
text_img_blurred = imgaussfilt(text_img, 1.5);

% 应用各种锐化方法
text_roberts = roberts_sharpen(text_img_blurred);
text_sobel = sobel_sharpen(text_img_blurred);
text_laplace = laplace_sharpen(text_img_blurred, laplace_kernel_4);
text_unsharp = unsharp_mask(text_img_blurred, 1.5);

% 显示文本增强效果
figure('Name', '图像锐化在文本增强中的应用', 'Position', [150, 150, 1200, 800]);

% 原始文本图像
subplot(2, 3, 1);
imshow(text_img);
title('原始文本图像');

% 模糊文本图像
subplot(2, 3, 2);
imshow(text_img_blurred);
title('模糊后的文本图像');

% Roberts算子增强
subplot(2, 3, 3);
imshow(text_roberts);
title('Roberts算子文本增强');

% Sobel算子增强
subplot(2, 3, 4);
imshow(text_sobel);
title('Sobel算子文本增强');

% Laplace算子增强
subplot(2, 3, 5);
imshow(text_laplace);
title('Laplace算子文本增强');

% 非锐化掩模增强
subplot(2, 3, 6);
imshow(text_unsharp);
title('非锐化掩模文本增强');

%% 11. 频率域锐化方法
fprintf('\n频率域锐化方法：高通滤波...\n');

% 创建理想高通滤波器
ideal_hp = ideal_highpass_filter(m, n, 30); % 截止频率30

% 创建巴特沃斯高通滤波器
butterworth_hp = butterworth_highpass_filter(m, n, 30, 2); % 截止频率30，阶数2

% 应用频率域滤波
img_freq_ideal = frequency_sharpen(original_img, ideal_hp);
img_freq_butterworth = frequency_sharpen(original_img, butterworth_hp);

% 显示频率域锐化结果
figure('Name', '频率域锐化方法', 'Position', [100, 100, 1200, 800]);

% 原始图像
subplot(2, 3, 1);
imshow(original_img);
title('原始图像');

% 理想高通滤波结果
subplot(2, 3, 2);
imshow(img_freq_ideal);
title('理想高通滤波锐化');

% 巴特沃斯高通滤波结果
subplot(2, 3, 3);
imshow(img_freq_butterworth);
title('巴特沃斯高通滤波锐化');

% 显示滤波器频率响应
subplot(2, 3, 4);
imshow(ideal_hp, []);
title('理想高通滤波器频率响应');
colorbar;

subplot(2, 3, 5);
imshow(butterworth_hp, []);
title('巴特沃斯高通滤波器频率响应');
colorbar;

% 对比空间域和频率域锐化效果
subplot(2, 3, 6);
sharpness_ideal = compute_sharpness(img_freq_ideal);
sharpness_butterworth = compute_sharpness(img_freq_butterworth);
sharpness_spatial = compute_sharpness(sobel_sharp);

bar(1:3, [sharpness_ideal, sharpness_butterworth, sharpness_spatial], ...
    'FaceColor', [0.6 0.6 0.9]);
set(gca, 'XTickLabel', {'理想高通', '巴特沃斯高通', 'Sobel算子'});
ylabel('锐化度指标');
title('频率域与空间域锐化效果对比');
grid on;

%% 12. 实验总结与效果评估
fprintf('\n实验总结与效果评估...\n');

% 计算各种锐化方法的综合评估指标
methods = {'Roberts', 'Sobel', 'Laplace(4)', 'Laplace(8)', '非锐化掩模', '理想高通', '巴特沃斯高通'};
sharpness_scores = zeros(length(methods), 1);
edge_preservation_scores = zeros(length(methods), 1);
noise_amplification_scores = zeros(length(methods), 1);

% 应用各种锐化方法到降噪后图像
sharpened_images = {
    roberts_sharpen(denoised_gaussian),
    sobel_sharpen(denoised_gaussian),
    laplace_sharpen(denoised_gaussian, laplace_kernel_4),
    laplace_sharpen(denoised_gaussian, laplace_kernel_8),
    unsharp_mask(denoised_gaussian, 1.0),
    frequency_sharpen(denoised_gaussian, ideal_hp),
    frequency_sharpen(denoised_gaussian, butterworth_hp)
};

% 计算各项指标
for i = 1:length(methods)
    sharpness_scores(i) = compute_sharpness(sharpened_images{i});
    edge_preservation_scores(i) = compute_edge_preservation(denoised_gaussian, sharpened_images{i});
    noise_amplification_scores(i) = compute_noise_amplification(denoised_gaussian, sharpened_images{i}, noisy_img);
end

% 归一化指标（0-100分）
sharpness_norm = 100 * (sharpness_scores - min(sharpness_scores)) / (max(sharpness_scores) - min(sharpness_scores));
edge_preservation_norm = 100 * (edge_preservation_scores - min(edge_preservation_scores)) / (max(edge_preservation_scores) - min(edge_preservation_scores));
noise_amplification_norm = 100 - 100 * (noise_amplification_scores - min(noise_amplification_scores)) / (max(noise_amplification_scores) - min(noise_amplification_scores));

% 综合评分（加权平均）
weights = [0.4, 0.3, 0.3]; % 锐化度、边缘保持、噪声放大的权重
composite_scores = sharpness_norm * weights(1) + ...
                   edge_preservation_norm * weights(2) + ...
                   noise_amplification_norm * weights(3);

% 显示综合评估结果
figure('Name', '锐化方法综合评估', 'Position', [150, 150, 1200, 600]);

% 各项指标对比
subplot(1, 3, 1);
hold on;
for i = 1:length(methods)
    plot([1, 2, 3], [sharpness_norm(i), edge_preservation_norm(i), noise_amplification_norm(i)], ...
         'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', methods{i});
end
xlim([0.5 3.5]);
set(gca, 'XTick', [1, 2, 3]);
set(gca, 'XTickLabel', {'锐化度', '边缘保持', '抗噪性'});
ylabel('评分 (0-100)');
title('各项指标对比');
legend('Location', 'best', 'NumColumns', 2);
grid on;

% 综合评分柱状图
subplot(1, 3, 2);
bar(1:length(methods), composite_scores, 'FaceColor', [0.6 0.8 0.8]);
set(gca, 'XTickLabel', methods);
ylabel('综合评分');
title('锐化方法综合评分');
grid on;
xtickangle(45);

% 添加数值标签
for i = 1:length(methods)
    text(i, composite_scores(i), sprintf('%.1f', composite_scores(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
end

% 雷达图展示（需要自定义函数）
subplot(1, 3, 3);
% 选择前4种方法展示
selected_methods = 1:4;
selected_scores = [sharpness_norm(selected_methods), ...
                   edge_preservation_norm(selected_methods), ...
                   noise_amplification_norm(selected_methods)];

% 简单柱状图替代雷达图
bar_data = [sharpness_norm(selected_methods), edge_preservation_norm(selected_methods), noise_amplification_norm(selected_methods)]';
bar(bar_data, 'grouped');
xlabel('评估指标');
ylabel('评分');
title('前4种方法指标对比');
set(gca, 'XTickLabel', {'锐化度', '边缘保持', '抗噪性'});
legend(methods(selected_methods), 'Location', 'best');
grid on;

%% 13. 实验结论
fprintf('\n实验完成！\n');
fprintf('================================================================\n');
fprintf('实验结论：\n');
fprintf('1. 图像锐化能够有效增强图像边缘和细节，提高图像清晰度。\n');
fprintf('2. 不同锐化方法有不同特点：\n');
fprintf('   - Roberts算子：简单快速，但对噪声敏感\n');
fprintf('   - Sobel算子：抗噪性较好，边缘定位准确\n');
fprintf('   - Laplace算子：增强所有方向的边缘，但可能产生双边缘\n');
fprintf('   - 非锐化掩模：可调节性强，效果自然\n');
fprintf('   - 频率域方法：理论完备，可设计复杂滤波器\n');
fprintf('3. 锐化前应先进行降噪处理，否则噪声会被放大。\n');
fprintf('4. 不同应用场景应选择不同的锐化方法：\n');
fprintf('   - 文本增强：Laplace算子或非锐化掩模\n');
fprintf('   - 医学图像：Sobel算子或频率域方法\n');
fprintf('   - 摄影后期：非锐化掩模\n');
fprintf('5. 锐化参数需要根据图像内容和需求进行调整。\n');
fprintf('================================================================\n');

%% ====================================================
%% 所有函数定义（已移动到文件末尾）
%% ====================================================

%% 1. Roberts算子锐化函数
function sharpened_img = roberts_sharpen(img)
    % Roberts算子锐化
    % 输入参数：
    %   img - 输入图像（灰度图像）
    % 输出参数：
    %   sharpened_img - 锐化后的图像
    
    % 转换为double类型
    img_double = double(img);
    
    % 定义Roberts算子
    Gx = [1, 0; 0, -1];
    Gy = [0, 1; -1, 0];
    
    % 计算梯度
    grad_x = conv2(img_double, Gx, 'same');
    grad_y = conv2(img_double, Gy, 'same');
    
    % 计算梯度幅值
    grad_mag = sqrt(grad_x.^2 + grad_y.^2);
    
    % 增强图像：原始图像 + α * 梯度幅值
    alpha = 0.5; % 增强系数
    sharpened = img_double + alpha * grad_mag;
    
    % 限制像素值范围并转换为uint8
    sharpened = max(0, min(255, sharpened));
    sharpened_img = uint8(sharpened);
end

%% 2. Sobel算子锐化函数
function sharpened_img = sobel_sharpen(img)
    % Sobel算子锐化
    % 输入参数：
    %   img - 输入图像
    % 输出参数：
    %   sharpened_img - 锐化后的图像
    
    % 转换为double类型
    img_double = double(img);
    
    % 定义Sobel算子
    Gx = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
    Gy = [1, 2, 1; 0, 0, 0; -1, -2, -1];
    
    % 计算梯度
    grad_x = conv2(img_double, Gx, 'same');
    grad_y = conv2(img_double, Gy, 'same');
    
    % 计算梯度幅值
    grad_mag = sqrt(grad_x.^2 + grad_y.^2);
    
    % 增强图像
    alpha = 0.3; % 增强系数
    sharpened = img_double + alpha * grad_mag;
    
    % 限制像素值范围并转换为uint8
    sharpened = max(0, min(255, sharpened));
    sharpened_img = uint8(sharpened);
end

%% 3. Sobel梯度计算函数
function [grad_mag, grad_dir] = sobel_gradient(img)
    % 计算Sobel梯度幅值和方向
    % 输入参数：
    %   img - 输入图像
    % 输出参数：
    %   grad_mag - 梯度幅值
    %   grad_dir - 梯度方向（弧度）
    
    % 转换为double类型
    img_double = double(img);
    
    % 定义Sobel算子
    Gx = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
    Gy = [1, 2, 1; 0, 0, 0; -1, -2, -1];
    
    % 计算梯度
    grad_x = conv2(img_double, Gx, 'same');
    grad_y = conv2(img_double, Gy, 'same');
    
    % 计算梯度幅值和方向
    grad_mag = sqrt(grad_x.^2 + grad_y.^2);
    grad_dir = atan2(grad_y, grad_x);
    
    % 归一化梯度幅值到0-255范围
    grad_mag = grad_mag / max(grad_mag(:)) * 255;
end

%% 4. Laplace算子锐化函数
function sharpened_img = laplace_sharpen(img, laplace_kernel)
    % Laplace算子锐化
    % 输入参数：
    %   img - 输入图像
    %   laplace_kernel - Laplace算子核
    % 输出参数：
    %   sharpened_img - 锐化后的图像
    
    % 转换为double类型
    img_double = double(img);
    
    % 应用Laplace算子
    laplace_response = conv2(img_double, laplace_kernel, 'same');
    
    % 增强图像：原始图像 - β * Laplace响应
    % 注意：对于中心为负的Laplace算子，应该用原始图像减去响应
    beta = 0.5; % 增强系数
    sharpened = img_double - beta * laplace_response;
    
    % 限制像素值范围并转换为uint8
    sharpened = max(0, min(255, sharpened));
    sharpened_img = uint8(sharpened);
end

%% 5. 非锐化掩模函数
function sharpened_img = unsharp_mask(img, k)
    % 非锐化掩模锐化
    % 输入参数：
    %   img - 输入图像
    %   k - 增强系数
    % 输出参数：
    %   sharpened_img - 锐化后的图像
    
    % 转换为double类型
    img_double = double(img);
    
    % 创建高斯低通滤波器（模糊图像）
    sigma = 1.5;
    h = fspecial('gaussian', 5, sigma);
    blurred = imfilter(img_double, h, 'replicate');
    
    % 计算细节（原始图像 - 模糊图像）
    detail = img_double - blurred;
    
    % 增强图像：原始图像 + k * 细节
    sharpened = img_double + k * detail;
    
    % 限制像素值范围并转换为uint8
    sharpened = max(0, min(255, sharpened));
    sharpened_img = uint8(sharpened);
end

%% 6. 边缘增强程度计算函数
function enhancement = compute_edge_enhancement(original, sharpened)
    % 计算边缘增强程度
    % 输入参数：
    %   original - 原始图像
    %   sharpened - 锐化后图像
    % 输出参数：
    %   enhancement - 边缘增强程度（百分比）
    
    % 计算原始图像的边缘强度（使用Sobel算子）
    [grad_mag_orig, ~] = sobel_gradient(original);
    edge_strength_orig = mean(grad_mag_orig(:));
    
    % 计算锐化后图像的边缘强度
    [grad_mag_sharp, ~] = sobel_gradient(sharpened);
    edge_strength_sharp = mean(grad_mag_sharp(:));
    
    % 计算增强程度
    enhancement = (edge_strength_sharp - edge_strength_orig) / edge_strength_orig * 100;
end

%% 7. 锐化度计算函数
function sharpness = compute_sharpness(img)
    % 计算图像锐化度
    % 输入参数：
    %   img - 输入图像
    % 输出参数：
    %   sharpness - 锐化度指标
    
    % 转换为double类型
    img_double = double(img);
    
    % 使用Laplace算子计算图像的高频成分
    laplace_kernel = [0, 1, 0; 1, -4, 1; 0, 1, 0];
    laplace_response = conv2(img_double, laplace_kernel, 'same');
    
    % 计算高频成分的方差作为锐化度指标
    sharpness = var(laplace_response(:));
end

%% 8. 边缘保持指标计算函数
function preservation = compute_edge_preservation(original, sharpened)
    % 计算边缘保持指标
    % 输入参数：
    %   original - 原始图像
    %   sharpened - 锐化后图像
    % 输出参数：
    %   preservation - 边缘保持指标（百分比）
    
    % 检测原始图像的边缘
    edge_orig = edge(original, 'canny');
    
    % 检测锐化后图像的边缘
    edge_sharp = edge(sharpened, 'canny');
    
    % 计算边缘相似度（Jaccard相似系数）
    intersection = sum(edge_orig(:) & edge_sharp(:));
    union = sum(edge_orig(:) | edge_sharp(:));
    
    if union == 0
        preservation = 0;
    else
        preservation = intersection / union * 100;
    end
end

%% 9. 噪声放大指标计算函数
function amplification = compute_noise_amplification(denoised, sharpened, noisy)
    % 计算噪声放大指标
    % 输入参数：
    %   denoised - 降噪后图像
    %   sharpened - 锐化后图像
    %   noisy - 噪声图像（用于参考）
    % 输出参数：
    %   amplification - 噪声放大指标（值越小越好）
    
    % 计算降噪图像的噪声成分
    noise_denoised = double(noisy) - double(denoised);
    
    % 计算锐化图像的噪声成分（近似）
    noise_sharpened = double(sharpened) - double(denoised);
    
    % 计算噪声放大程度
    noise_power_denoised = mean(noise_denoised(:).^2);
    noise_power_sharpened = mean(noise_sharpened(:).^2);
    
    if noise_power_denoised == 0
        amplification = 0;
    else
        amplification = noise_power_sharpened / noise_power_denoised;
    end
end

%% 10. 理想高通滤波器生成函数
function H = ideal_highpass_filter(m, n, cutoff)
    % 生成理想高通滤波器
    % 输入参数：
    %   m, n - 滤波器尺寸
    %   cutoff - 截止频率（半径）
    % 输出参数：
    %   H - 滤波器频率响应
    
    % 创建频率网格
    [U, V] = meshgrid(1:n, 1:m);
    
    % 计算频率坐标（中心化）
    U = U - floor(n/2) - 1;
    V = V - floor(m/2) - 1;
    
    % 计算频率距离
    D = sqrt(U.^2 + V.^2);
    
    % 创建理想高通滤波器
    H = double(D > cutoff);
    
    % 将零频率分量移回中心
    H = ifftshift(H);
end

%% 11. 巴特沃斯高通滤波器生成函数
function H = butterworth_highpass_filter(m, n, cutoff, order)
    % 生成巴特沃斯高通滤波器
    % 输入参数：
    %   m, n - 滤波器尺寸
    %   cutoff - 截止频率（半径）
    %   order - 滤波器阶数
    % 输出参数：
    %   H - 滤波器频率响应
    
    % 创建频率网格
    [U, V] = meshgrid(1:n, 1:m);
    
    % 计算频率坐标（中心化）
    U = U - floor(n/2) - 1;
    V = V - floor(m/2) - 1;
    
    % 计算频率距离
    D = sqrt(U.^2 + V.^2);
    
    % 创建巴特沃斯高通滤波器
    H = 1 ./ (1 + (cutoff ./ (D + eps)).^(2 * order));
    
    % 将零频率分量移回中心
    H = ifftshift(H);
end

%% 12. 频率域锐化函数
function sharpened_img = frequency_sharpen(img, H)
    % 频率域锐化
    % 输入参数：
    %   img - 输入图像
    %   H - 滤波器频率响应
    % 输出参数：
    %   sharpened_img - 锐化后的图像
    
    % 转换为double类型
    img_double = double(img);
    
    % 计算图像的傅里叶变换
    F = fft2(img_double);
    
    % 应用滤波器
    G = F .* H;
    
    % 计算逆傅里叶变换
    sharpened_freq = real(ifft2(G));
    
    % 限制像素值范围并转换为uint8
    sharpened_freq = max(0, min(255, sharpened_freq));
    sharpened_img = uint8(sharpened_freq);
end