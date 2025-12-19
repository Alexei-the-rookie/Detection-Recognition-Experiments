%% 实验八：图像平滑实验
clear; close all; clc;
fprintf('=== 图像平滑实验 ===\n\n');

%% 1. 实验原理说明
fprintf('实验原理：\n');
fprintf('平滑滤波的目的是消除或尽量减少噪声，改善图像质量。\n');
fprintf('平滑本质上是低通滤波，允许低频信号通过，阻止高频噪声信号。\n');
fprintf('但图像边缘也处于高频部分，平滑处理往往会对图像边缘造成损坏。\n\n');

%% 2. 加载原始图像
fprintf('加载原始图像...\n');
original_img = imread("D:\University Work And Life\Detection & Identifying Technology\IMG_20231208_165218.jpg");

% 确保图像是灰度图像
if size(original_img, 3) == 3
    original_img = rgb2gray(original_img);
end

% 调整图像大小以加快处理速度
original_img = imresize(original_img, [256, 256]);

% 显示图像信息
[m, n] = size(original_img);
fprintf('图像尺寸: %d × %d 像素\n', m, n);

%% 3. 显示原始图像和频率分析
figure('Name', '原始图像与频率分析', 'Position', [100, 100, 1200, 600]);

% 原始图像
subplot(1, 2, 1);
imshow(original_img);
title('原始图像');

% 原始图像的傅里叶频谱
subplot(1, 2, 2);
F = fft2(double(original_img));
F_shift = fftshift(F);
magnitude_spectrum = log(abs(F_shift) + 1);
imagesc(magnitude_spectrum);
colormap(gray);
title('原始图像傅里叶频谱');
colorbar;
axis image;

%% 4. 添加不同类型和强度的噪声
fprintf('\n添加不同类型和强度的噪声...\n');

% 低强度高斯噪声
noise_gaussian_low = imnoise(original_img, 'gaussian', 0, 0.005);

% 高强度高斯噪声
noise_gaussian_high = imnoise(original_img, 'gaussian', 0, 0.02);

% 低密度椒盐噪声
noise_salt_pepper_low = imnoise(original_img, 'salt & pepper', 0.02);

% 高密度椒盐噪声
noise_salt_pepper_high = imnoise(original_img, 'salt & pepper', 0.1);

% 显示噪声图像
figure('Name', '不同类型和强度的噪声图像', 'Position', [150, 150, 1200, 800]);

subplot(2, 2, 1);
imshow(noise_gaussian_low);
title('低强度高斯噪声 (方差=0.005)');

subplot(2, 2, 2);
imshow(noise_gaussian_high);
title('高强度高斯噪声 (方差=0.02)');

subplot(2, 2, 3);
imshow(noise_salt_pepper_low);
title('低密度椒盐噪声 (密度=0.02)');

subplot(2, 2, 4);
imshow(noise_salt_pepper_high);
title('高密度椒盐噪声 (密度=0.1)');

%% 5. 均值滤波（邻域平均滤波）
fprintf('\n使用均值滤波进行图像平滑...\n');

% 定义不同的窗口大小
mean_window_sizes = [3, 5, 7, 9];

% 对低强度高斯噪声应用均值滤波
figure('Name', '均值滤波效果（低强度高斯噪声）', 'Position', [100, 100, 1200, 800]);

for i = 1:length(mean_window_sizes)
    window_size = mean_window_sizes(i);
    
    % 使用自定义均值滤波函数
    filtered_mean = mean_filter_custom(noise_gaussian_low, window_size);
    
    % 计算PSNR
    psnr_value = psnr(filtered_mean, original_img);
    
    % 显示结果
    subplot(2, 4, i);
    imshow(filtered_mean);
    title(sprintf('均值滤波 %d×%d\nPSNR: %.2f dB', window_size, window_size, psnr_value));
    
    % 显示边缘检测结果
    subplot(2, 4, i+4);
    edge_original = edge(original_img, 'canny');
    edge_filtered = edge(filtered_mean, 'canny');
    edge_similarity = sum(edge_original(:) & edge_filtered(:)) / sum(edge_original(:)) * 100;
    
    imshowpair(edge_original, edge_filtered, 'montage');
    title(sprintf('边缘对比\n相似度: %.1f%%', edge_similarity));
end

% 对高密度椒盐噪声应用均值滤波
figure('Name', '均值滤波效果（高密度椒盐噪声）', 'Position', [150, 150, 1200, 800]);

for i = 1:length(mean_window_sizes)
    window_size = mean_window_sizes(i);
    
    % 使用自定义均值滤波函数
    filtered_mean = mean_filter_custom(noise_salt_pepper_high, window_size);
    
    % 计算PSNR
    psnr_value = psnr(filtered_mean, original_img);
    
    % 显示结果
    subplot(2, 4, i);
    imshow(filtered_mean);
    title(sprintf('均值滤波 %d×%d\nPSNR: %.2f dB', window_size, window_size, psnr_value));
    
    % 显示残差图像（噪声去除效果）
    subplot(2, 4, i+4);
    residual = double(noise_salt_pepper_high) - double(filtered_mean);
    imshow(residual, []);
    title('去除的噪声');
    colorbar;
end

%% 6. 高斯滤波
fprintf('\n使用高斯滤波进行图像平滑...\n');

% 定义不同的标准差
sigma_values = [0.5, 1.0, 1.5, 2.0];

% 对低强度高斯噪声应用高斯滤波
figure('Name', '高斯滤波效果（低强度高斯噪声）', 'Position', [100, 100, 1200, 800]);

for i = 1:length(sigma_values)
    sigma = sigma_values(i);
    
    % 创建高斯滤波器
    kernel_size = 2 * ceil(3 * sigma) + 1; % 确保窗口足够大
    gaussian_filter = fspecial('gaussian', kernel_size, sigma);
    
    % 应用高斯滤波
    filtered_gaussian = imfilter(noise_gaussian_low, gaussian_filter, 'replicate');
    
    % 计算PSNR
    psnr_value = psnr(filtered_gaussian, original_img);
    
    % 显示结果
    subplot(2, 4, i);
    imshow(filtered_gaussian);
    title(sprintf('高斯滤波 σ=%.1f\nPSNR: %.2f dB', sigma, psnr_value));
    
    % 显示高斯滤波器
    subplot(2, 4, i+4);
    imagesc(gaussian_filter);
    title(sprintf('高斯滤波器 %d×%d', kernel_size, kernel_size));
    colorbar;
    axis image;
end

% 对高强度高斯噪声应用高斯滤波
figure('Name', '高斯滤波效果（高强度高斯噪声）', 'Position', [150, 150, 1200, 800]);

for i = 1:length(sigma_values)
    sigma = sigma_values(i);
    
    % 创建高斯滤波器
    kernel_size = 2 * ceil(3 * sigma) + 1;
    gaussian_filter = fspecial('gaussian', kernel_size, sigma);
    
    % 应用高斯滤波
    filtered_gaussian = imfilter(noise_gaussian_high, gaussian_filter, 'replicate');
    
    % 计算PSNR
    psnr_value = psnr(filtered_gaussian, original_img);
    
    % 显示结果
    subplot(2, 4, i);
    imshow(filtered_gaussian);
    title(sprintf('高斯滤波 σ=%.1f\nPSNR: %.2f dB', sigma, psnr_value));
    
    % 显示边缘检测结果
    subplot(2, 4, i+4);
    edge_original = edge(original_img, 'canny');
    edge_filtered = edge(filtered_gaussian, 'canny');
    edge_similarity = sum(edge_original(:) & edge_filtered(:)) / sum(edge_original(:)) * 100;
    
    imshowpair(edge_original, edge_filtered, 'montage');
    title(sprintf('边缘对比\n相似度: %.1f%%', edge_similarity));
end

%% 7. 中值滤波（对比实验）
fprintf('\n使用中值滤波进行图像平滑（对比实验）...\n');

% 对高密度椒盐噪声应用中值滤波
median_window_sizes = [3, 5, 7, 9];

figure('Name', '中值滤波效果（高密度椒盐噪声）', 'Position', [100, 100, 1200, 800]);

for i = 1:length(median_window_sizes)
    window_size = median_window_sizes(i);
    
    % 使用自定义中值滤波函数
    filtered_median = median_filter_custom(noise_salt_pepper_high, window_size);
    
    % 计算PSNR
    psnr_value = psnr(filtered_median, original_img);
    
    % 显示结果
    subplot(2, 4, i);
    imshow(filtered_median);
    title(sprintf('中值滤波 %d×%d\nPSNR: %.2f dB', window_size, window_size, psnr_value));
    
    % 显示残差图像
    subplot(2, 4, i+4);
    residual = double(noise_salt_pepper_high) - double(filtered_median);
    imshow(residual, []);
    title('去除的噪声');
    colorbar;
end

%% 8. 不同平滑方法的对比分析
fprintf('\n不同平滑方法的对比分析...\n');

% 对低强度高斯噪声应用不同平滑方法
test_img = noise_gaussian_low;
window_size = 5;
sigma = 1.0;

% 应用不同的平滑方法
filtered_mean = mean_filter_custom(test_img, window_size);
filtered_gaussian = imgaussfilt(test_img, sigma);
filtered_median = median_filter_custom(test_img, window_size);

% 计算性能指标
psnr_mean = psnr(filtered_mean, original_img);
psnr_gaussian = psnr(filtered_gaussian, original_img);
psnr_median = psnr(filtered_median, original_img);

mse_mean = immse(filtered_mean, original_img);
mse_gaussian = immse(filtered_gaussian, original_img);
mse_median = immse(filtered_median, original_img);

% 计算边缘保持指标
edge_original = edge(original_img, 'canny');
edge_mean = edge(filtered_mean, 'canny');
edge_gaussian = edge(filtered_gaussian, 'canny');
edge_median = edge(filtered_median, 'canny');

edge_preservation_mean = sum(edge_original(:) & edge_mean(:)) / sum(edge_original(:)) * 100;
edge_preservation_gaussian = sum(edge_original(:) & edge_gaussian(:)) / sum(edge_original(:)) * 100;
edge_preservation_median = sum(edge_original(:) & edge_median(:)) / sum(edge_original(:)) * 100;

% 显示对比结果
figure('Name', '不同平滑方法对比（低强度高斯噪声）', 'Position', [150, 150, 1200, 800]);

% 原始噪声图像
subplot(3, 5, 1);
imshow(test_img);
title('高斯噪声图像');

% 均值滤波结果
subplot(3, 5, 2);
imshow(filtered_mean);
title(sprintf('均值滤波\nPSNR: %.2f dB', psnr_mean));

% 高斯滤波结果
subplot(3, 5, 3);
imshow(filtered_gaussian);
title(sprintf('高斯滤波\nPSNR: %.2f dB', psnr_gaussian));

% 中值滤波结果
subplot(3, 5, 4);
imshow(filtered_median);
title(sprintf('中值滤波\nPSNR: %.2f dB', psnr_median));

% 性能对比柱状图 - PSNR
subplot(3, 5, 5);
bar(1:3, [psnr_mean, psnr_gaussian, psnr_median], 'FaceColor', [0.6 0.8 0.6]);
set(gca, 'XTickLabel', {'均值滤波', '高斯滤波', '中值滤波'});
ylabel('PSNR (dB)');
title('PSNR对比');
grid on;

% 显示边缘检测结果 - 均值滤波
subplot(3, 5, 6);
imshowpair(edge_original, edge_mean, 'montage');
title(sprintf('均值滤波边缘\n保持度: %.1f%%', edge_preservation_mean));

% 显示边缘检测结果 - 高斯滤波
subplot(3, 5, 7);
imshowpair(edge_original, edge_gaussian, 'montage');
title(sprintf('高斯滤波边缘\n保持度: %.1f%%', edge_preservation_gaussian));

% 显示边缘检测结果 - 中值滤波
subplot(3, 5, 8);
imshowpair(edge_original, edge_median, 'montage');
title(sprintf('中值滤波边缘\n保持度: %.1f%%', edge_preservation_median));

% 性能对比柱状图 - MSE
subplot(3, 5, 9);
bar(1:3, [mse_mean, mse_gaussian, mse_median], 'FaceColor', [0.8 0.6 0.6]);
set(gca, 'XTickLabel', {'均值滤波', '高斯滤波', '中值滤波'});
ylabel('MSE');
title('MSE对比');
grid on;

% 性能对比柱状图 - 边缘保持度
subplot(3, 5, 10);
bar(1:3, [edge_preservation_mean, edge_preservation_gaussian, edge_preservation_median], ...
    'FaceColor', [0.6 0.6 0.8]);
set(gca, 'XTickLabel', {'均值滤波', '高斯滤波', '中值滤波'});
ylabel('边缘保持度 (%)');
title('边缘保持对比');
ylim([0 100]);
grid on;

%% 9. 平滑对图像边缘的影响分析
fprintf('\n分析平滑对图像边缘的影响...\n');

% 创建测试图像：包含清晰边缘
edge_test_img = zeros(256, 256, 'uint8');
edge_test_img(100:156, 100:156) = 255; % 白色方块

% 添加噪声
edge_test_noisy = imnoise(edge_test_img, 'gaussian', 0, 0.01);

% 应用不同的平滑方法
edge_filtered_mean = mean_filter_custom(edge_test_noisy, 7);
edge_filtered_gaussian = imgaussfilt(edge_test_noisy, 1.5);
edge_filtered_median = median_filter_custom(edge_test_noisy, 7);

% 计算边缘扩散（边缘宽度）
edge_width_original = compute_edge_width(edge_test_img);
edge_width_mean = compute_edge_width(edge_filtered_mean);
edge_width_gaussian = compute_edge_width(edge_filtered_gaussian);
edge_width_median = compute_edge_width(edge_filtered_median);

% 显示边缘影响分析结果
figure('Name', '平滑对图像边缘的影响分析', 'Position', [150, 150, 1200, 800]);

% 原始边缘图像
subplot(3, 4, 1);
imshow(edge_test_img);
title('原始边缘图像');

% 噪声边缘图像
subplot(3, 4, 2);
imshow(edge_test_noisy);
title('加噪边缘图像');

% 均值滤波结果
subplot(3, 4, 3);
imshow(edge_filtered_mean);
title(sprintf('均值滤波\n边缘宽度: %.2f像素', edge_width_mean));

% 高斯滤波结果
subplot(3, 4, 4);
imshow(edge_filtered_gaussian);
title(sprintf('高斯滤波\n边缘宽度: %.2f像素', edge_width_gaussian));

% 中值滤波结果
subplot(3, 4, 5);
imshow(edge_filtered_median);
title(sprintf('中值滤波\n边缘宽度: %.2f像素', edge_width_median));

% 显示边缘剖面（水平方向，通过图像中心）
center_row = 128;
profile_original = double(edge_test_img(center_row, :));
profile_noisy = double(edge_test_noisy(center_row, :));
profile_mean = double(edge_filtered_mean(center_row, :));
profile_gaussian = double(edge_filtered_gaussian(center_row, :));
profile_median = double(edge_filtered_median(center_row, :));

subplot(3, 4, [6, 7, 8]);
plot(1:256, profile_original, 'k-', 'LineWidth', 2, 'DisplayName', '原始图像');
hold on;
plot(1:256, profile_noisy, 'r-', 'LineWidth', 1, 'DisplayName', '噪声图像');
plot(1:256, profile_mean, 'b-', 'LineWidth', 1, 'DisplayName', '均值滤波');
plot(1:256, profile_gaussian, 'g-', 'LineWidth', 1, 'DisplayName', '高斯滤波');
plot(1:256, profile_median, 'm-', 'LineWidth', 1, 'DisplayName', '中值滤波');
hold off;
xlabel('像素位置');
ylabel('灰度值');
title('边缘剖面分析（第128行）');
legend('Location', 'best');
grid on;

% 边缘宽度对比
subplot(3, 4, [9, 10]);
bar(1:4, [edge_width_original, edge_width_mean, edge_width_gaussian, edge_width_median], ...
    'FaceColor', [0.8 0.6 0.8]);
set(gca, 'XTickLabel', {'原始图像', '均值滤波', '高斯滤波', '中值滤波'});
ylabel('边缘宽度 (像素)');
title('边缘宽度对比');
grid on;

% 边缘锐度计算
edge_sharpness_original = compute_edge_sharpness(edge_test_img);
edge_sharpness_mean = compute_edge_sharpness(edge_filtered_mean);
edge_sharpness_gaussian = compute_edge_sharpness(edge_filtered_gaussian);
edge_sharpness_median = compute_edge_sharpness(edge_filtered_median);

subplot(3, 4, [11, 12]);
bar(1:4, [edge_sharpness_original, edge_sharpness_mean, edge_sharpness_gaussian, edge_sharpness_median], ...
    'FaceColor', [0.6 0.8 0.8]);
set(gca, 'XTickLabel', {'原始图像', '均值滤波', '高斯滤波', '中值滤波'});
ylabel('边缘锐度');
title('边缘锐度对比');
grid on;

%% 10. 平滑滤波器频率响应分析
fprintf('\n分析平滑滤波器的频率响应...\n');

% 创建不同的滤波器
% 均值滤波器 5×5
mean_kernel_5 = ones(5, 5) / 25;

% 高斯滤波器 5×5，σ=1
gaussian_kernel_5 = fspecial('gaussian', 5, 1);

% 计算频率响应
fft_size = 64;
mean_freq = abs(fftshift(fft2(mean_kernel_5, fft_size, fft_size)));
gaussian_freq = abs(fftshift(fft2(gaussian_kernel_5, fft_size, fft_size)));

% 显示频率响应
figure('Name', '平滑滤波器频率响应分析', 'Position', [100, 100, 1200, 600]);

% 均值滤波器频率响应
subplot(2, 3, 1);
imagesc(log(mean_freq + 1));
colormap(jet);
title('均值滤波器频率响应（对数）');
colorbar;
axis image;

subplot(2, 3, 2);
mesh(mean_freq);
title('均值滤波器频率响应（3D）');
xlabel('水平频率');
ylabel('垂直频率');
zlabel('幅度');

% 高斯滤波器频率响应
subplot(2, 3, 4);
imagesc(log(gaussian_freq + 1));
colormap(jet);
title('高斯滤波器频率响应（对数）');
colorbar;
axis image;

subplot(2, 3, 5);
mesh(gaussian_freq);
title('高斯滤波器频率响应（3D）');
xlabel('水平频率');
ylabel('垂直频率');
zlabel('幅度');

% 频率响应剖面（通过中心）
center_freq = fft_size/2;
freq_profile_mean = mean_freq(center_freq, :);
freq_profile_gaussian = gaussian_freq(center_freq, :);

subplot(2, 3, [3, 6]);
plot(1:fft_size, freq_profile_mean, 'b-', 'LineWidth', 2, 'DisplayName', '均值滤波器');
hold on;
plot(1:fft_size, freq_profile_gaussian, 'r-', 'LineWidth', 2, 'DisplayName', '高斯滤波器');
hold off;
xlabel('频率');
ylabel('幅度');
title('频率响应剖面（通过中心）');
legend('Location', 'best');
grid on;

%% 11. 自适应平滑方法
fprintf('\n使用自适应平滑方法...\n');

% 创建自适应平滑函数（基于局部方差）
adaptive_filtered = adaptive_smoothing(noise_gaussian_low, 3, 0.01);

% 显示自适应平滑结果
figure('Name', '自适应平滑方法效果', 'Position', [150, 150, 1000, 600]);

% 原始噪声图像
subplot(2, 3, 1);
imshow(noise_gaussian_low);
title('高斯噪声图像');

% 自适应平滑结果
subplot(2, 3, 2);
imshow(adaptive_filtered);
psnr_adaptive = psnr(adaptive_filtered, original_img);
title(sprintf('自适应平滑\nPSNR: %.2f dB', psnr_adaptive));

% 固定均值滤波结果（对比）
fixed_mean = mean_filter_custom(noise_gaussian_low, 3);
subplot(2, 3, 3);
imshow(fixed_mean);
psnr_fixed = psnr(fixed_mean, original_img);
title(sprintf('固定均值滤波\nPSNR: %.2f dB', psnr_fixed));

% 局部方差图
subplot(2, 3, 4);
local_variance = compute_local_variance(noise_gaussian_low, 3);
imagesc(local_variance);
colormap(jet);
title('局部方差图');
colorbar;

% 自适应权重图
subplot(2, 3, 5);
adaptive_weights = compute_adaptive_weights(local_variance, 0.01);
imagesc(adaptive_weights);
colormap(jet);
title('自适应权重图');
colorbar;

% 边缘保持对比
subplot(2, 3, 6);
edge_adaptive = edge(adaptive_filtered, 'canny');
edge_fixed = edge(fixed_mean, 'canny');
edge_original = edge(original_img, 'canny');

edge_preservation_adaptive = sum(edge_original(:) & edge_adaptive(:)) / sum(edge_original(:)) * 100;
edge_preservation_fixed = sum(edge_original(:) & edge_fixed(:)) / sum(edge_original(:)) * 100;

bar(1:2, [edge_preservation_fixed, edge_preservation_adaptive], 'FaceColor', [0.8 0.6 0.6]);
set(gca, 'XTickLabel', {'固定滤波', '自适应滤波'});
ylabel('边缘保持度 (%)');
title('边缘保持对比');
ylim([0 100]);
grid on;

%% 12. 平滑方法的综合评估
fprintf('\n平滑方法的综合评估...\n');

% 测试不同噪声类型下的平滑效果
noise_types = {'低强度高斯', '高强度高斯', '低密度椒盐', '高密度椒盐'};
noise_images = {noise_gaussian_low, noise_gaussian_high, noise_salt_pepper_low, noise_salt_pepper_high};

% 平滑方法
methods = {'均值滤波', '高斯滤波', '中值滤波', '自适应平滑'};

% 初始化结果矩阵
psnr_results = zeros(length(noise_types), length(methods));
edge_preservation_results = zeros(length(noise_types), length(methods));

% 计算各项指标
for i = 1:length(noise_types)
    current_noise = noise_images{i};
    
    % 均值滤波
    filtered_mean = mean_filter_custom(current_noise, 5);
    psnr_results(i, 1) = psnr(filtered_mean, original_img);
    edge_mean = edge(filtered_mean, 'canny');
    edge_preservation_results(i, 1) = sum(edge_original(:) & edge_mean(:)) / sum(edge_original(:)) * 100;
    
    % 高斯滤波
    filtered_gaussian = imgaussfilt(current_noise, 1.0);
    psnr_results(i, 2) = psnr(filtered_gaussian, original_img);
    edge_gaussian = edge(filtered_gaussian, 'canny');
    edge_preservation_results(i, 2) = sum(edge_original(:) & edge_gaussian(:)) / sum(edge_original(:)) * 100;
    
    % 中值滤波
    filtered_median = median_filter_custom(current_noise, 5);
    psnr_results(i, 3) = psnr(filtered_median, original_img);
    edge_median = edge(filtered_median, 'canny');
    edge_preservation_results(i, 3) = sum(edge_original(:) & edge_median(:)) / sum(edge_original(:)) * 100;
    
    % 自适应平滑
    filtered_adaptive = adaptive_smoothing(current_noise, 3, 0.01);
    psnr_results(i, 4) = psnr(filtered_adaptive, original_img);
    edge_adaptive = edge(filtered_adaptive, 'canny');
    edge_preservation_results(i, 4) = sum(edge_original(:) & edge_adaptive(:)) / sum(edge_original(:)) * 100;
end

% 显示综合评估结果
figure('Name', '平滑方法综合评估', 'Position', [150, 150, 1200, 600]);

% PSNR结果热图
subplot(1, 3, 1);
imagesc(psnr_results);
colormap(jet);
colorbar;
title('PSNR结果热图 (dB)');
xlabel('平滑方法');
ylabel('噪声类型');
set(gca, 'XTick', 1:length(methods));
set(gca, 'XTickLabel', methods);
set(gca, 'YTick', 1:length(noise_types));
set(gca, 'YTickLabel', noise_types);

% 添加数值标签
for i = 1:length(noise_types)
    for j = 1:length(methods)
        text(j, i, sprintf('%.1f', psnr_results(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white', 'FontSize', 9);
    end
end

% 边缘保持结果热图
subplot(1, 3, 2);
imagesc(edge_preservation_results);
colormap(jet);
colorbar;
title('边缘保持结果热图 (%)');
xlabel('平滑方法');
ylabel('噪声类型');
set(gca, 'XTick', 1:length(methods));
set(gca, 'XTickLabel', methods);
set(gca, 'YTick', 1:length(noise_types));
set(gca, 'YTickLabel', noise_types);

% 添加数值标签
for i = 1:length(noise_types)
    for j = 1:length(methods)
        text(j, i, sprintf('%.1f', edge_preservation_results(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white', 'FontSize', 9);
    end
end

% 综合评分（PSNR和边缘保持的加权平均）
weights = [0.6, 0.4]; % PSNR权重0.6，边缘保持权重0.4

% 归一化
psnr_norm = (psnr_results - min(psnr_results(:))) / (max(psnr_results(:)) - min(psnr_results(:)));
edge_norm = (edge_preservation_results - min(edge_preservation_results(:))) / ...
            (max(edge_preservation_results(:)) - min(edge_preservation_results(:)));

composite_scores = psnr_norm * weights(1) + edge_norm * weights(2);

subplot(1, 3, 3);
imagesc(composite_scores);
colormap(jet);
colorbar;
title('综合评分热图');
xlabel('平滑方法');
ylabel('噪声类型');
set(gca, 'XTick', 1:length(methods));
set(gca, 'XTickLabel', methods);
set(gca, 'YTick', 1:length(noise_types));
set(gca, 'YTickLabel', noise_types);

% 添加数值标签
for i = 1:length(noise_types)
    for j = 1:length(methods)
        text(j, i, sprintf('%.2f', composite_scores(i, j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white', 'FontSize', 9);
    end
end

%% 13. 实验结论
fprintf('\n实验完成！\n');
fprintf('================================================================\n');
fprintf('实验结论：\n');
fprintf('1. 平滑滤波能有效抑制噪声，但会模糊图像边缘。\n');
fprintf('2. 不同平滑方法的特点：\n');
fprintf('   - 均值滤波：简单快速，但边缘保持差\n');
fprintf('   - 高斯滤波：平滑效果好，边缘过渡自然\n');
fprintf('   - 中值滤波：对椒盐噪声效果好，边缘保持较好\n');
fprintf('   - 自适应平滑：根据局部特性调整，平衡去噪和边缘保持\n');
fprintf('3. 平滑滤波器本质上是低通滤波器，会衰减图像高频成分。\n');
fprintf('4. 平滑参数选择：\n');
fprintf('   - 窗口大小：越大去噪效果越好，但边缘越模糊\n');
fprintf('   - 高斯σ：越大平滑程度越高\n');
fprintf('5. 噪声类型影响平滑方法选择：\n');
fprintf('   - 高斯噪声：高斯滤波效果较好\n');
fprintf('   - 椒盐噪声：中值滤波效果较好\n');
fprintf('================================================================\n');

%% ====================================================
%% 所有函数定义（已移动到文件末尾）
%% ====================================================

%% 1. 均值滤波函数
function filtered_img = mean_filter_custom(img, window_size)
    % 均值滤波函数
    % 输入参数：
    %   img - 输入图像
    %   window_size - 窗口大小（奇数）
    % 输出参数：
    %   filtered_img - 滤波后的图像
    
    % 检查窗口大小是否为奇数
    if mod(window_size, 2) == 0
        error('窗口大小必须是奇数');
    end
    
    % 转换为double类型
    img_double = double(img);
    [rows, cols] = size(img_double);
    
    % 创建输出图像
    filtered_img = zeros(rows, cols);
    
    % 计算填充大小
    pad_size = floor(window_size / 2);
    
    % 对图像进行边界填充（使用镜像填充）
    img_padded = padarray(img_double, [pad_size, pad_size], 'symmetric');
    
    % 计算均值
    for i = 1:rows
        for j = 1:cols
            % 提取当前窗口
            window = img_padded(i:i+window_size-1, j:j+window_size-1);
            
            % 计算均值
            mean_value = mean(window(:));
            
            % 赋值给输出图像
            filtered_img(i, j) = mean_value;
        end
    end
    
    % 转换回uint8类型
    filtered_img = uint8(filtered_img);
end

%% 2. 中值滤波函数
function filtered_img = median_filter_custom(img, window_size)
    % 中值滤波函数
    % 输入参数：
    %   img - 输入图像
    %   window_size - 窗口大小（奇数）
    % 输出参数：
    %   filtered_img - 滤波后的图像
    
    % 检查窗口大小是否为奇数
    if mod(window_size, 2) == 0
        error('窗口大小必须是奇数');
    end
    
    % 转换为double类型
    img_double = double(img);
    [rows, cols] = size(img_double);
    
    % 创建输出图像
    filtered_img = zeros(rows, cols);
    
    % 计算填充大小
    pad_size = floor(window_size / 2);
    
    % 对图像进行边界填充（使用镜像填充）
    img_padded = padarray(img_double, [pad_size, pad_size], 'symmetric');
    
    % 计算中值
    for i = 1:rows
        for j = 1:cols
            % 提取当前窗口
            window = img_padded(i:i+window_size-1, j:j+window_size-1);
            
            % 计算中值
            median_value = median(window(:));
            
            % 赋值给输出图像
            filtered_img(i, j) = median_value;
        end
    end
    
    % 转换回uint8类型
    filtered_img = uint8(filtered_img);
end

%% 3. 边缘宽度计算函数
function width = compute_edge_width(img)
    % 计算图像边缘宽度
    % 输入参数：
    %   img - 输入图像（包含边缘）
    % 输出参数：
    %   width - 边缘宽度（像素）
    
    % 转换为double类型
    img_double = double(img);
    
    % 计算图像梯度（使用Sobel算子）
    [Gx, Gy] = gradient(img_double);
    grad_mag = sqrt(Gx.^2 + Gy.^2);
    
    % 计算梯度直方图
    [counts, values] = histcounts(grad_mag(:), 100);
    
    % 找到梯度峰值
    [~, max_idx] = max(counts);
    peak_value = values(max_idx);
    
    % 计算梯度大于峰值一半的像素比例
    threshold = peak_value / 2;
    edge_pixels = grad_mag > threshold;
    
    % 计算边缘区域面积
    edge_area = sum(edge_pixels(:));
    
    % 计算边缘长度（使用边缘检测）
    edge_binary = edge(img, 'canny');
    edge_length = sum(edge_binary(:));
    
    % 计算平均边缘宽度
    if edge_length > 0
        width = edge_area / edge_length;
    else
        width = 0;
    end
end

%% 4. 边缘锐度计算函数
function sharpness = compute_edge_sharpness(img)
    % 计算图像边缘锐度
    % 输入参数：
    %   img - 输入图像
    % 输出参数：
    %   sharpness - 边缘锐度指标
    
    % 转换为double类型
    img_double = double(img);
    
    % 计算图像梯度（使用Sobel算子）
    [Gx, Gy] = gradient(img_double);
    grad_mag = sqrt(Gx.^2 + Gy.^2);
    
    % 计算梯度平均值作为锐度指标
    sharpness = mean(grad_mag(:));
end

%% 5. 局部方差计算函数
function variance_map = compute_local_variance(img, window_size)
    % 计算图像局部方差
    % 输入参数：
    %   img - 输入图像
    %   window_size - 窗口大小（奇数）
    % 输出参数：
    %   variance_map - 局部方差图
    
    % 转换为double类型
    img_double = double(img);
    [rows, cols] = size(img_double);
    
    % 创建输出图像
    variance_map = zeros(rows, cols);
    
    % 计算填充大小
    pad_size = floor(window_size / 2);
    
    % 对图像进行边界填充
    img_padded = padarray(img_double, [pad_size, pad_size], 'replicate');
    
    % 计算局部方差
    for i = 1:rows
        for j = 1:cols
            % 提取当前窗口
            window = img_padded(i:i+window_size-1, j:j+window_size-1);
            
            % 计算方差
            variance_map(i, j) = var(window(:));
        end
    end
end

%% 6. 自适应权重计算函数
function weights = compute_adaptive_weights(variance_map, noise_variance)
    % 计算自适应权重
    % 输入参数：
    %   variance_map - 局部方差图
    %   noise_variance - 噪声方差估计
    % 输出参数：
    %   weights - 自适应权重图
    
    % 避免除以零
    variance_map = max(variance_map, eps);
    
    % 计算权重：噪声方差与局部方差的比值
    weights = noise_variance ./ variance_map;
    
    % 限制权重范围（0-1）
    weights = min(weights, 1);
    weights = max(weights, 0);
end

%% 7. 自适应平滑函数
function filtered_img = adaptive_smoothing(img, window_size, noise_variance)
    % 自适应平滑函数
    % 输入参数：
    %   img - 输入图像
    %   window_size - 窗口大小（奇数）
    %   noise_variance - 噪声方差估计
    % 输出参数：
    %   filtered_img - 自适应平滑后的图像
    
    % 转换为double类型
    img_double = double(img);
    [rows, cols] = size(img_double);
    
    % 计算局部方差
    variance_map = compute_local_variance(img, window_size);
    
    % 计算自适应权重
    weights = compute_adaptive_weights(variance_map, noise_variance);
    
    % 创建输出图像
    filtered_img = zeros(rows, cols);
    
    % 计算填充大小
    pad_size = floor(window_size / 2);
    
    % 对图像和权重进行边界填充
    img_padded = padarray(img_double, [pad_size, pad_size], 'replicate');
    weights_padded = padarray(weights, [pad_size, pad_size], 'replicate');
    
    % 自适应平滑
    for i = 1:rows
        for j = 1:cols
            % 提取当前窗口
            img_window = img_padded(i:i+window_size-1, j:j+window_size-1);
            weight_window = weights_padded(i:i+window_size-1, j:j+window_size-1);
            
            % 计算加权平均
            weighted_sum = sum(img_window(:) .* weight_window(:));
            weight_sum = sum(weight_window(:));
            
            if weight_sum > 0
                filtered_img(i, j) = weighted_sum / weight_sum;
            else
                filtered_img(i, j) = img_double(i, j);
            end
        end
    end
    
    % 转换回uint8类型
    filtered_img = uint8(filtered_img);
end