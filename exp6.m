%% 实验六：中值滤波实验
clear; close all; clc;
fprintf('=== 中值滤波实验 ===\n\n');

%% 1. 实验原理说明
fprintf('实验原理：\n');
fprintf('中值滤波是一种非线性信号处理方法，属于统计排序滤波器。\n');
fprintf('它将每个像素点的灰度值设置为该点邻域窗口内所有像素点灰度值的中值。\n');
fprintf('中值滤波对椒盐噪声、脉冲噪声有良好的滤波效果，能保持图像边缘特性。\n\n');

%% 2. 加载原始图像
fprintf('加载原始图像...\n');
original_img = imread("D:\University Work And Life\Detection & Identifying Technology\IMG_20231208_165218.jpg");

% 确保图像是灰度图像
if size(original_img, 3) == 3
    original_img = rgb2gray(original_img);
end

% 显示原始图像信息
[m, n] = size(original_img);
fprintf('图像尺寸: %d × %d 像素\n', m, n);

%% 3. 添加不同类型的噪声
fprintf('\n添加不同类型的噪声...\n');

% 添加椒盐噪声
noise_density = 0.05; % 噪声密度
img_salt_pepper = imnoise(original_img, 'salt & pepper', noise_density);

% 添加高斯噪声
gaussian_noise_level = 0.02; % 噪声水平
img_gaussian = imnoise(original_img, 'gaussian', 0, gaussian_noise_level);

% 添加脉冲噪声（随机黑白点）
img_impulse = original_img;
impulse_mask = rand(m, n) < noise_density/2;
img_impulse(impulse_mask) = 0; % 黑色脉冲
impulse_mask = rand(m, n) < noise_density/2;
img_impulse(impulse_mask) = 255; % 白色脉冲

% 显示噪声图像
figure('Name', '原始图像与噪声图像', 'Position', [100, 100, 1200, 800]);

subplot(2, 2, 1);
imshow(original_img);
title('原始图像');

subplot(2, 2, 2);
imshow(img_salt_pepper);
title(sprintf('椒盐噪声 (密度=%.3f)', noise_density));

subplot(2, 2, 3);
imshow(img_gaussian);
title(sprintf('高斯噪声 (水平=%.3f)', gaussian_noise_level));

subplot(2, 2, 4);
imshow(img_impulse);
title(sprintf('脉冲噪声 (密度=%.3f)', noise_density));

%% 4. 实现中值滤波函数
fprintf('\n实现中值滤波函数...\n');

%% 5. 不同窗口大小的中值滤波效果
fprintf('\n测试不同窗口大小的中值滤波效果...\n');

% 测试不同窗口大小
window_sizes = [3, 5, 7, 9];

figure('Name', '不同窗口大小的中值滤波效果（椒盐噪声）', 'Position', [150, 150, 1400, 800]);

for i = 1:length(window_sizes)
    window_size = window_sizes(i);
    
    % 应用中值滤波
    filtered_img = median_filter(img_salt_pepper, window_size);
    
    % 计算PSNR
    psnr_value = psnr(filtered_img, original_img);
    
    % 显示结果
    subplot(2, length(window_sizes), i);
    imshow(filtered_img);
    title(sprintf('窗口大小 %d×%d\nPSNR: %.2f dB', window_size, window_size, psnr_value));
    
    % 计算并显示残差图像
    subplot(2, length(window_sizes), i+length(window_sizes));
    residual = double(img_salt_pepper) - double(filtered_img);
    imshow(residual, []);
    title('残差图像（噪声-滤波）');
    colorbar;
end

%% 6. 中值滤波与均值滤波对比
fprintf('\n中值滤波与均值滤波对比...\n');

% 使用椒盐噪声图像进行对比
window_size = 3;

% 中值滤波
median_filtered = median_filter(img_salt_pepper, window_size);

% 均值滤波（使用MATLAB内置函数）
mean_filtered = uint8(filter2(fspecial('average', window_size), double(img_salt_pepper)));

% 计算性能指标
psnr_median = psnr(median_filtered, original_img);
psnr_mean = psnr(mean_filtered, original_img);

mse_median = immse(median_filtered, original_img);
mse_mean = immse(mean_filtered, original_img);

% 显示对比结果
figure('Name', '中值滤波与均值滤波对比', 'Position', [100, 100, 1200, 800]);

% 椒盐噪声图像
subplot(2, 4, 1);
imshow(img_salt_pepper);
title('椒盐噪声图像');

subplot(2, 4, 5);
imhist(img_salt_pepper);
title('椒盐噪声直方图');
xlim([0 255]);

% 中值滤波结果
subplot(2, 4, 2);
imshow(median_filtered);
title(sprintf('中值滤波 %d×%d\nPSNR: %.2f dB', window_size, window_size, psnr_median));

subplot(2, 4, 6);
imhist(median_filtered);
title('中值滤波后直方图');
xlim([0 255]);

% 均值滤波结果
subplot(2, 4, 3);
imshow(mean_filtered);
title(sprintf('均值滤波 %d×%d\nPSNR: %.2f dB', window_size, window_size, psnr_mean));

subplot(2, 4, 7);
imhist(mean_filtered);
title('均值滤波后直方图');
xlim([0 255]);

% 残差对比
subplot(2, 4, [4, 8]);
hold on;
box_data = [mse_median, mse_mean];
bar(1:2, box_data, 'FaceColor', [0.6 0.6 0.8]);
set(gca, 'XTickLabel', {'中值滤波MSE', '均值滤波MSE'});
ylabel('均方误差 (MSE)');
title('滤波性能对比');
text(1, mse_median, sprintf('%.4f', mse_median), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(2, mse_mean, sprintf('%.4f', mse_mean), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
grid on;

%% 7. 中值滤波对不同类型噪声的处理效果
fprintf('\n测试中值滤波对不同类型噪声的处理效果...\n');

% 对三种噪声图像应用中值滤波
window_size = 3;

median_salt_pepper = median_filter(img_salt_pepper, window_size);
median_gaussian = median_filter(img_gaussian, window_size);
median_impulse = median_filter(img_impulse, window_size);

% 计算PSNR
psnr_sp = psnr(median_salt_pepper, original_img);
psnr_gaussian = psnr(median_gaussian, original_img);
psnr_impulse = psnr(median_impulse, original_img);

% 显示结果
figure('Name', '中值滤波对不同类型噪声的处理效果', 'Position', [150, 150, 1400, 1000]);

% 椒盐噪声处理
subplot(3, 4, 1);
imshow(img_salt_pepper);
title('椒盐噪声图像');

subplot(3, 4, 2);
imshow(median_salt_pepper);
title(sprintf('中值滤波后\nPSNR: %.2f dB', psnr_sp));

subplot(3, 4, 3);
% 计算边缘（使用Sobel算子）
edge_orig = edge(original_img, 'sobel');
edge_filtered = edge(median_salt_pepper, 'sobel');
edge_similarity = sum(edge_orig(:) & edge_filtered(:)) / sum(edge_orig(:)) * 100;
imshowpair(edge_orig, edge_filtered, 'montage');
title(sprintf('边缘对比\n相似度: %.1f%%', edge_similarity));

subplot(3, 4, 4);
noise_removed = double(img_salt_pepper) - double(median_salt_pepper);
imshow(noise_removed, []);
title('去除的噪声');
colorbar;

% 高斯噪声处理
subplot(3, 4, 5);
imshow(img_gaussian);
title('高斯噪声图像');

subplot(3, 4, 6);
imshow(median_gaussian);
title(sprintf('中值滤波后\nPSNR: %.2f dB', psnr_gaussian));

subplot(3, 4, 7);
edge_filtered_gauss = edge(median_gaussian, 'sobel');
edge_similarity_gauss = sum(edge_orig(:) & edge_filtered_gauss(:)) / sum(edge_orig(:)) * 100;
imshowpair(edge_orig, edge_filtered_gauss, 'montage');
title(sprintf('边缘对比\n相似度: %.1f%%', edge_similarity_gauss));

subplot(3, 4, 8);
noise_removed_gauss = double(img_gaussian) - double(median_gaussian);
imshow(noise_removed_gauss, []);
title('去除的噪声');
colorbar;

% 脉冲噪声处理
subplot(3, 4, 9);
imshow(img_impulse);
title('脉冲噪声图像');

subplot(3, 4, 10);
imshow(median_impulse);
title(sprintf('中值滤波后\nPSNR: %.2f dB', psnr_impulse));

subplot(3, 4, 11);
edge_filtered_impulse = edge(median_impulse, 'sobel');
edge_similarity_impulse = sum(edge_orig(:) & edge_filtered_impulse(:)) / sum(edge_orig(:)) * 100;
imshowpair(edge_orig, edge_filtered_impulse, 'montage');
title(sprintf('边缘对比\n相似度: %.1f%%', edge_similarity_impulse));

subplot(3, 4, 12);
noise_removed_impulse = double(img_impulse) - double(median_impulse);
imshow(noise_removed_impulse, []);
title('去除的噪声');
colorbar;

%% 8. 不同形状滤波窗口的效果
fprintf('\n测试不同形状滤波窗口的效果...\n');

% 定义不同形状的窗口
% 3x3方形窗口
window_square = ones(3, 3);

% 十字形窗口
window_cross = [0, 1, 0;
                1, 1, 1;
                0, 1, 0];

% 圆形窗口（近似）
window_circle = [0, 1, 0;
                 1, 1, 1;
                 0, 1, 0];

% 对不同噪声图像应用不同形状窗口的中值滤波
img_test = img_salt_pepper;

% 方形窗口
filtered_square = median_filter_custom(img_test, window_square);

% 十字形窗口
filtered_cross = median_filter_custom(img_test, window_cross);

% 计算性能指标
psnr_square = psnr(filtered_square, original_img);
psnr_cross = psnr(filtered_cross, original_img);

% 显示结果
figure('Name', '不同形状滤波窗口的效果', 'Position', [200, 200, 1200, 600]);

subplot(2, 3, 1);
imshow(img_test);
title('椒盐噪声图像');

subplot(2, 3, 2);
imagesc(window_square);
title('方形窗口');
axis equal;
axis off;
colormap(gray);

subplot(2, 3, 3);
imagesc(window_cross);
title('十字形窗口');
axis equal;
axis off;
colormap(gray);

subplot(2, 3, 4);
imshow(filtered_square);
title(sprintf('方形窗口滤波\nPSNR: %.2f dB', psnr_square));

subplot(2, 3, 5);
imshow(filtered_cross);
title(sprintf('十字形窗口滤波\nPSNR: %.2f dB', psnr_cross));

subplot(2, 3, 6);
% 边缘保持性能对比
edge_square = edge(filtered_square, 'sobel');
edge_cross = edge(filtered_cross, 'sobel');
edge_similarity_square = sum(edge_orig(:) & edge_square(:)) / sum(edge_orig(:)) * 100;
edge_similarity_cross = sum(edge_orig(:) & edge_cross(:)) / sum(edge_orig(:)) * 100;

bar([1, 2], [edge_similarity_square, edge_similarity_cross], 'FaceColor', [0.6 0.8 0.6]);
set(gca, 'XTickLabel', {'方形窗口', '十字形窗口'});
ylabel('边缘相似度 (%)');
title('边缘保持性能对比');
ylim([0 100]);
grid on;

%% 9. 中值滤波的迭代应用
fprintf('\n测试中值滤波的迭代应用效果...\n');

% 应用多次中值滤波
num_iterations = 3;
window_size = 3;

iterative_filtered = img_salt_pepper;
psnr_values = zeros(num_iterations, 1);
mse_values = zeros(num_iterations, 1);

for iter = 1:num_iterations
    iterative_filtered = median_filter(iterative_filtered, window_size);
    psnr_values(iter) = psnr(iterative_filtered, original_img);
    mse_values(iter) = immse(iterative_filtered, original_img);
end

% 显示迭代结果
figure('Name', '中值滤波的迭代应用', 'Position', [150, 150, 1200, 800]);

% 显示每次迭代的结果
for iter = 1:num_iterations
    % 重新计算每次迭代（为了显示）
    temp_img = img_salt_pepper;
    for i = 1:iter
        temp_img = median_filter(temp_img, window_size);
    end
    
    subplot(2, num_iterations, iter);
    imshow(temp_img);
    title(sprintf('第 %d 次迭代\nPSNR: %.2f dB', iter, psnr_values(iter)));
    
    % 计算边缘
    edge_iter = edge(temp_img, 'sobel');
    subplot(2, num_iterations, iter+num_iterations);
    imshowpair(edge_orig, edge_iter, 'montage');
    title(sprintf('边缘对比 (迭代 %d)', iter));
end

% 绘制PSNR和MSE随迭代次数的变化
figure('Name', '滤波性能随迭代次数的变化', 'Position', [200, 200, 800, 400]);

subplot(1, 2, 1);
plot(1:num_iterations, psnr_values, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('迭代次数');
ylabel('PSNR (dB)');
title('PSNR随迭代次数的变化');
grid on;

subplot(1, 2, 2);
plot(1:num_iterations, mse_values, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('迭代次数');
ylabel('MSE');
title('MSE随迭代次数的变化');
grid on;

%% 10. 中值滤波在边缘检测前的预处理应用
fprintf('\n测试中值滤波在边缘检测前的预处理效果...\n');

% 对噪声图像进行边缘检测（无预处理）
edge_noisy = edge(img_salt_pepper, 'sobel');

% 先进行中值滤波，再进行边缘检测
filtered_for_edge = median_filter(img_salt_pepper, 3);
edge_filtered_first = edge(filtered_for_edge, 'sobel');

% 计算边缘检测性能
edge_detection_noisy = sum(edge_noisy(:) & edge_orig(:)) / sum(edge_orig(:)) * 100;
edge_detection_filtered = sum(edge_filtered_first(:) & edge_orig(:)) / sum(edge_orig(:)) * 100;

% 显示结果
figure('Name', '中值滤波在边缘检测前的预处理', 'Position', [100, 100, 1200, 600]);

subplot(2, 3, 1);
imshow(img_salt_pepper);
title('椒盐噪声图像');

subplot(2, 3, 2);
imshow(filtered_for_edge);
title('中值滤波预处理后');

subplot(2, 3, 3);
bar([1, 2], [edge_detection_noisy, edge_detection_filtered], 'FaceColor', [0.8 0.6 0.6]);
set(gca, 'XTickLabel', {'无预处理', '中值滤波预处理'});
ylabel('边缘检测准确率 (%)');
title('边缘检测性能对比');
ylim([0 100]);
grid on;

subplot(2, 3, 4);
imshow(edge_noisy);
title(sprintf('直接边缘检测\n准确率: %.1f%%', edge_detection_noisy));

subplot(2, 3, 5);
imshow(edge_filtered_first);
title(sprintf('滤波后边缘检测\n准确率: %.1f%%', edge_detection_filtered));

subplot(2, 3, 6);
imshowpair(edge_orig, edge_filtered_first, 'falsecolor');
title('边缘对比（颜色叠加）');

%% 11. 中值滤波的计算复杂度分析
fprintf('\n分析中值滤波的计算复杂度...\n');

% 测试不同窗口大小的计算时间
window_sizes_test = [3, 5, 7, 9, 11, 15];
time_results = zeros(length(window_sizes_test), 1);

% 使用较小的测试图像以加快计算
test_img_small = imresize(img_salt_pepper, [128, 128]);

fprintf('窗口大小 | 处理时间(秒)\n');
fprintf('--------|-------------\n');

for i = 1:length(window_sizes_test)
    window_size_test = window_sizes_test(i);
    
    % 计时
    tic;
    for repeat = 1:10  % 重复10次以获得更稳定的时间
        temp_result = median_filter(test_img_small, window_size_test);
    end
    elapsed_time = toc / 10;  % 平均时间
    
    time_results(i) = elapsed_time;
    fprintf('%d×%d    | %.4f\n', window_size_test, window_size_test, elapsed_time);
end

% 绘制计算时间与窗口大小的关系
figure('Name', '中值滤波计算复杂度分析', 'Position', [150, 150, 800, 400]);

subplot(1, 2, 1);
plot(window_sizes_test, time_results, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('窗口大小 (N×N)');
ylabel('处理时间 (秒)');
title('处理时间 vs 窗口大小');
grid on;

% 理论复杂度分析
% 中值滤波的复杂度约为 O(N² log N) 其中N是窗口大小
% 绘制理论曲线
subplot(1, 2, 2);
n_squared_log_n = (window_sizes_test.^2) .* log(window_sizes_test);
n_squared_log_n_normalized = n_squared_log_n / max(n_squared_log_n) * max(time_results);

plot(window_sizes_test, time_results, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '实际时间');
hold on;
plot(window_sizes_test, n_squared_log_n_normalized, 'b--', 'LineWidth', 2, 'DisplayName', '理论趋势 (N²logN)');
xlabel('窗口大小 (N×N)');
ylabel('归一化时间');
title('实际时间 vs 理论复杂度');
legend('Location', 'northwest');
grid on;

%% 12. 综合实验结果展示
fprintf('\n展示综合实验结果...\n');

figure('Name', '中值滤波综合实验结果', 'Position', [100, 100, 1400, 1000]);

% 原始图像和噪声图像
subplot(3, 4, 1);
imshow(original_img);
title('原始图像');

subplot(3, 4, 2);
imshow(img_salt_pepper);
title('椒盐噪声图像');

% 最佳中值滤波结果（3x3窗口）
best_filtered = median_filter(img_salt_pepper, 3);
psnr_best = psnr(best_filtered, original_img);

subplot(3, 4, 3);
imshow(best_filtered);
title(sprintf('中值滤波 (3×3)\nPSNR: %.2f dB', psnr_best));

% 噪声去除效果
subplot(3, 4, 4);
noise_removed_best = double(img_salt_pepper) - double(best_filtered);
imshow(noise_removed_best, []);
title('去除的噪声');
colorbar;

% 边缘保持效果
subplot(3, 4, 5);
edge_original = edge(original_img, 'canny');
imshow(edge_original);
title('原始图像边缘');

subplot(3, 4, 6);
edge_noisy_img = edge(img_salt_pepper, 'canny');
imshow(edge_noisy_img);
title('噪声图像边缘');

subplot(3, 4, 7);
edge_filtered_img = edge(best_filtered, 'canny');
imshow(edge_filtered_img);
title('滤波后图像边缘');

% 边缘相似度计算
edge_similarity_best = sum(edge_original(:) & edge_filtered_img(:)) / sum(edge_original(:)) * 100;

subplot(3, 4, 8);
imshowpair(edge_original, edge_filtered_img, 'falsecolor');
title(sprintf('边缘对比\n相似度: %.1f%%', edge_similarity_best));

% 不同噪声类型处理效果对比
subplot(3, 4, 9);
imshow(median_filter(img_gaussian, 3));
title('高斯噪声 + 中值滤波');

subplot(3, 4, 10);
imshow(median_filter(img_impulse, 3));
title('脉冲噪声 + 中值滤波');

% 与均值滤波对比
subplot(3, 4, 11);
mean_filtered_3x3 = uint8(filter2(fspecial('average', 3), double(img_salt_pepper)));
imshow(mean_filtered_3x3);
psnr_mean_3x3 = psnr(mean_filtered_3x3, original_img);
title(sprintf('均值滤波 (3×3)\nPSNR: %.2f dB', psnr_mean_3x3));

% 性能对比柱状图
subplot(3, 4, 12);
hold on;
bar(1, psnr_best, 'FaceColor', [0.2 0.6 0.8], 'BarWidth', 0.6);
bar(2, psnr_mean_3x3, 'FaceColor', [0.8 0.4 0.4], 'BarWidth', 0.6);
ylabel('PSNR (dB)');
title('中值滤波 vs 均值滤波');
set(gca, 'XTick', [1, 2]);
set(gca, 'XTickLabel', {'中值滤波', '均值滤波'});
grid on;

% 添加数值标签
text(1, psnr_best, sprintf('%.2f', psnr_best), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(2, psnr_mean_3x3, sprintf('%.2f', psnr_mean_3x3), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);

%% 13. 实验总结
fprintf('\n实验完成！\n');
fprintf('================================================================\n');
fprintf('实验结论：\n');
fprintf('1. 中值滤波能有效去除椒盐噪声和脉冲噪声，保持图像边缘。\n');
fprintf('2. 窗口大小影响滤波效果：小窗口保持细节但去噪不彻底，大窗口去噪彻底但可能模糊细节。\n');
fprintf('3. 中值滤波对高斯噪声效果有限，不如均值滤波有效。\n');
fprintf('4. 中值滤波是非线性滤波器，计算复杂度高于线性滤波器。\n');
fprintf('5. 中值滤波能很好地保持图像边缘，适合作为边缘检测的预处理步骤。\n');
fprintf('6. 多次迭代应用中值滤波可以进一步去除噪声，但可能导致图像过度平滑。\n');
fprintf('================================================================\n');

%% ====================================================
%% 所有函数定义（已移动到文件末尾）
%% ====================================================

%% 1. 标准中值滤波函数（方形窗口）
function filtered_img = median_filter(img, window_size)
    % 中值滤波函数
    % 输入参数：
    %   img - 输入图像（灰度图像）
    %   window_size - 窗口大小（奇数，如3, 5, 7等）
    % 输出参数：
    %   filtered_img - 滤波后的图像
    
    % 检查输入参数
    if mod(window_size, 2) == 0
        error('窗口大小必须是奇数');
    end
    
    % 转换为double类型以便计算
    img_double = double(img);
    [rows, cols] = size(img_double);
    
    % 创建输出图像
    filtered_img = zeros(rows, cols);
    
    % 计算填充大小
    pad_size = floor(window_size / 2);
    
    % 对图像进行边界填充（使用镜像填充）
    img_padded = padarray(img_double, [pad_size, pad_size], 'symmetric');
    
    % 应用中值滤波
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

%% 2. 自定义窗口形状的中值滤波函数
function filtered_img = median_filter_custom(img, window_mask)
    % 自定义窗口形状的中值滤波函数
    % 输入参数：
    %   img - 输入图像
    %   window_mask - 窗口掩码（二维矩阵，1表示包含在窗口中，0表示排除）
    % 输出参数：
    %   filtered_img - 滤波后的图像
    
    % 检查窗口掩码
    if sum(window_mask(:) == 1) == 0
        error('窗口掩码必须至少包含一个1');
    end
    
    % 转换为double类型
    img_double = double(img);
    [rows, cols] = size(img_double);
    [w_rows, w_cols] = size(window_mask);
    
    % 确保窗口大小是奇数
    if mod(w_rows, 2) == 0 || mod(w_cols, 2) == 0
        error('窗口掩码的行数和列数都必须是奇数');
    end
    
    % 创建输出图像
    filtered_img = zeros(rows, cols);
    
    % 计算填充大小
    pad_rows = floor(w_rows / 2);
    pad_cols = floor(w_cols / 2);
    
    % 对图像进行边界填充（使用镜像填充）
    img_padded = padarray(img_double, [pad_rows, pad_cols], 'symmetric');
    
    % 获取窗口掩码中为1的索引
    [mask_rows, mask_cols] = find(window_mask == 1);
    num_pixels = length(mask_rows);
    
    % 应用中值滤波
    for i = 1:rows
        for j = 1:cols
            % 收集窗口内的像素值（仅考虑掩码为1的位置）
            window_values = zeros(num_pixels, 1);
            
            for k = 1:num_pixels
                row_idx = i + mask_rows(k) - 1;
                col_idx = j + mask_cols(k) - 1;
                window_values(k) = img_padded(row_idx, col_idx);
            end
            
            % 计算中值
            median_value = median(window_values);
            
            % 赋值给输出图像
            filtered_img(i, j) = median_value;
        end
    end
    
    % 转换回uint8类型
    filtered_img = uint8(filtered_img);
end

%% 3. 快速中值滤波函数（使用分离行列方法）
function filtered_img = fast_median_filter(img, window_size)
    % 快速中值滤波函数（行列分离方法）
    % 输入参数：
    %   img - 输入图像
    %   window_size - 窗口大小（奇数）
    % 输出参数：
    %   filtered_img - 滤波后的图像
    
    % 检查输入参数
    if mod(window_size, 2) == 0
        error('窗口大小必须是奇数');
    end
    
    % 转换为double类型
    img_double = double(img);
    [rows, cols] = size(img_double);
    
    % 第一步：对每一行进行水平方向的中值滤波
    temp_img = zeros(rows, cols);
    pad_size = floor(window_size / 2);
    
    % 水平方向滤波
    for i = 1:rows
        row_data = img_double(i, :);
        
        % 对行进行边界填充
        row_padded = padarray(row_data, [0, pad_size], 'symmetric');
        
        for j = 1:cols
            % 提取当前窗口
            window = row_padded(j:j+window_size-1);
            
            % 计算中值
            median_value = median(window);
            
            % 赋值给临时图像
            temp_img(i, j) = median_value;
        end
    end
    
    % 第二步：对每一列进行垂直方向的中值滤波
    filtered_img = zeros(rows, cols);
    
    % 垂直方向滤波
    for j = 1:cols
        col_data = temp_img(:, j)';
        
        % 对列进行边界填充
        col_padded = padarray(col_data, [0, pad_size], 'symmetric');
        
        for i = 1:rows
            % 提取当前窗口
            window = col_padded(i:i+window_size-1);
            
            % 计算中值
            median_value = median(window);
            
            % 赋值给输出图像
            filtered_img(i, j) = median_value;
        end
    end
    
    % 转换回uint8类型
    filtered_img = uint8(filtered_img);
end