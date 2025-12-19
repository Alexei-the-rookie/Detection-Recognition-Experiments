%% 实验四：直方图统计实验
clear; close all; clc;
fprintf('=== 直方图统计实验 ===\n\n');

%% 1. 图像加载与预处理
% 使用多种内置图像进行对比
fprintf('加载图像...\n');

% 加载多幅图像进行对比
img_names = {'cameraman.tif', 'pout.tif', 'coins.png', 'rice.png', 'peppers.png'};
img_titles = {'Cameraman', 'Pout', 'Coins', 'Rice', 'Peppers'};
images = cell(1, length(img_names));

for i = 1:length(img_names)
    try
        img = imread(img_names{i});
        % 转换为灰度图像
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        images{i} = img;
    catch
        fprintf('无法加载图像: %s\n', img_names{i});
        images{i} = zeros(256, 256, 'uint8'); % 创建空白图像占位
    end
end

%% 2. 计算并显示各图像的直方图
fprintf('\n计算并显示各图像的直方图...\n');

figure('Name', '原始图像与直方图对比', 'Position', [100, 100, 1400, 800]);

for i = 1:length(images)
    % 显示原始图像
    subplot(2, length(images), i);
    imshow(images{i});
    title(img_titles{i}, 'FontSize', 10);
    
    % 显示直方图
    subplot(2, length(images), i + length(images));
    [counts, gray_levels] = custom_histogram(images{i});
    bar(gray_levels, counts, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
    title('直方图', 'FontSize', 10);
    xlabel('灰度级');
    ylabel('像素数');
    xlim([0 255]);
    grid on;
    
    % 在直方图上标注统计信息
    total_pixels = numel(images{i});
    mean_value = mean(double(images{i}(:)));
    std_value = std(double(images{i}(:)));
    text(50, max(counts)*0.9, sprintf('均值: %.1f', mean_value), 'FontSize', 8);
    text(50, max(counts)*0.8, sprintf('标准差: %.1f', std_value), 'FontSize', 8);
end

%% 3. 直方图均衡化
fprintf('\n进行直方图均衡化...\n');

% 选择一幅图像进行详细分析
selected_img = images{1}; % cameraman
img_eq = histeq(selected_img);

% 使用自定义直方图均衡化函数
img_eq_custom = custom_histeq(selected_img);

figure('Name', '直方图均衡化对比', 'Position', [150, 150, 1200, 800]);

% 原始图像和直方图
subplot(2, 3, 1);
imshow(selected_img);
title('原始图像');

subplot(2, 3, 4);
[counts_orig, gray_orig] = custom_histogram(selected_img);
bar(gray_orig, counts_orig, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('原始直方图');
xlabel('灰度级');
ylabel('像素数');
grid on;

% MATLAB内置histeq结果
subplot(2, 3, 2);
imshow(img_eq);
title('MATLAB histeq');

subplot(2, 3, 5);
[counts_eq, gray_eq] = custom_histogram(img_eq);
bar(gray_eq, counts_eq, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('均衡化直方图(histeq)');
xlabel('灰度级');
ylabel('像素数');
grid on;

% 自定义均衡化结果
subplot(2, 3, 3);
imshow(img_eq_custom);
title('自定义均衡化');

subplot(2, 3, 6);
[counts_eq_custom, gray_eq_custom] = custom_histogram(img_eq_custom);
bar(gray_eq_custom, counts_eq_custom, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('均衡化直方图(自定义)');
xlabel('灰度级');
ylabel('像素数');
grid on;

%% 4. 直方图匹配（规定化）
fprintf('\n进行直方图匹配（规定化）...\n');

% 选择两幅图像进行直方图匹配
source_img = images{1}; % cameraman
target_img = images{2}; % pout

% 执行直方图匹配
img_matched = custom_histmatch(source_img, target_img);

figure('Name', '直方图匹配（规定化）', 'Position', [200, 200, 1200, 600]);

% 源图像
subplot(2, 4, 1);
imshow(source_img);
title('源图像');

subplot(2, 4, 5);
[counts_source, ~] = custom_histogram(source_img);
bar(0:255, counts_source, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('源直方图');
xlim([0 255]);
grid on;

% 目标图像
subplot(2, 4, 2);
imshow(target_img);
title('目标图像');

subplot(2, 4, 6);
[counts_target, ~] = custom_histogram(target_img);
bar(0:255, counts_target, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('目标直方图');
xlim([0 255]);
grid on;

% 匹配后的图像
subplot(2, 4, 3);
imshow(img_matched);
title('匹配后图像');

subplot(2, 4, 7);
[counts_matched, ~] = custom_histogram(img_matched);
bar(0:255, counts_matched, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('匹配后直方图');
xlim([0 255]);
grid on;

% 累积分布函数对比
subplot(2, 4, [4, 8]);
hold on;

% 计算累积分布函数
cdf_source = cumsum(counts_source) / sum(counts_source);
cdf_target = cumsum(counts_target) / sum(counts_target);
cdf_matched = cumsum(counts_matched) / sum(counts_matched);

plot(0:255, cdf_source, 'b-', 'LineWidth', 2, 'DisplayName', '源图像CDF');
plot(0:255, cdf_target, 'r-', 'LineWidth', 2, 'DisplayName', '目标图像CDF');
plot(0:255, cdf_matched, 'g-', 'LineWidth', 2, 'DisplayName', '匹配后CDF');

title('累积分布函数对比');
xlabel('灰度级');
ylabel('累积概率');
legend('Location', 'best');
grid on;

%% 5. 直方图统计分析
fprintf('\n进行直方图统计分析...\n');

% 计算各种统计特征
figure('Name', '直方图统计分析', 'Position', [100, 100, 1000, 800]);

for i = 1:min(4, length(images))
    current_img = images{i};
    
    % 计算直方图
    [counts, gray_levels] = custom_histogram(current_img);
    
    % 计算统计特征
    img_double = double(current_img(:));
    stats = compute_histogram_stats(img_double);
    
    % 显示图像
    subplot(4, 4, (i-1)*4 + 1);
    imshow(current_img);
    title(img_titles{i}, 'FontSize', 10);
    
    % 显示直方图
    subplot(4, 4, (i-1)*4 + 2);
    bar(gray_levels, counts, 'BarWidth', 1, 'FaceColor', [0.3 0.3 0.3]);
    xlim([0 255]);
    title('直方图', 'FontSize', 10);
    grid on;
    
    % 显示统计特征
    subplot(4, 4, [(i-1)*4 + 3, (i-1)*4 + 4]);
    axis off;
    
    text(0.1, 0.9, sprintf('图像: %s', img_titles{i}), 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.8, sprintf('总像素数: %d', stats.total_pixels), 'FontSize', 10);
    text(0.1, 0.7, sprintf('均值: %.2f', stats.mean_value), 'FontSize', 10);
    text(0.1, 0.6, sprintf('中位数: %.2f', stats.median_value), 'FontSize', 10);
    text(0.1, 0.5, sprintf('标准差: %.2f', stats.std_value), 'FontSize', 10);
    text(0.1, 0.4, sprintf('方差: %.2f', stats.variance), 'FontSize', 10);
    text(0.1, 0.3, sprintf('偏度: %.4f', stats.skewness), 'FontSize', 10);
    text(0.1, 0.2, sprintf('峰度: %.4f', stats.kurtosis), 'FontSize', 10);
    
    % 添加直方图分布类型判断
    if stats.skewness > 1
        dist_type = '右偏分布';
    elseif stats.skewness < -1
        dist_type = '左偏分布';
    else
        dist_type = '近似对称分布';
    end
    text(0.1, 0.1, sprintf('分布类型: %s', dist_type), 'FontSize', 10, 'Color', 'r');
end

%% 6. 直方图分割阈值分析
fprintf('\n进行直方图分割阈值分析...\n');

% 使用大津法（Otsu）计算最佳阈值
test_img = images{3}; % coins图像适合阈值分割
[threshold, binary_img] = otsu_threshold(test_img);

% 显示分割结果
figure('Name', '直方图分割阈值分析', 'Position', [150, 150, 1000, 600]);

% 原始图像
subplot(2, 3, 1);
imshow(test_img);
title('原始图像');

% 直方图与阈值
subplot(2, 3, 2);
[counts, gray_levels] = custom_histogram(test_img);
bar(gray_levels, counts, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
hold on;
line([threshold threshold], [0 max(counts)], 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--');
title(sprintf('直方图与Otsu阈值\n阈值 = %.1f', threshold));
xlabel('灰度级');
ylabel('像素数');
grid on;

% 二值图像
subplot(2, 3, 3);
imshow(binary_img);
title('二值分割结果');

% 尝试不同阈值
thresholds = [50, 100, 150, 200];
for i = 1:length(thresholds)
    subplot(2, 3, i+3);
    binary_temp = test_img > thresholds(i);
    imshow(binary_temp);
    title(sprintf('阈值 = %d', thresholds(i)));
end

%% 7. 直方图比较和相似度度量
fprintf('\n进行直方图比较和相似度度量...\n');

% 比较不同图像的直方图
figure('Name', '直方图比较', 'Position', [200, 200, 1200, 600]);

% 选择几幅图像进行比较
compare_indices = [1, 2, 3, 4]; % 比较前4幅图像

% 计算直方图
histograms = cell(1, length(compare_indices));
for i = 1:length(compare_indices)
    idx = compare_indices(i);
    [counts, ~] = custom_histogram(images{idx});
    histograms{i} = counts / sum(counts); % 归一化
end

% 显示直方图比较
for i = 1:length(compare_indices)
    subplot(2, length(compare_indices), i);
    bar(0:255, histograms{i}, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.8]);
    title(sprintf('%s 直方图', img_titles{compare_indices(i)}));
    xlim([0 255]);
    grid on;
end

% 计算直方图相似度
similarity_matrix = zeros(length(compare_indices));
methods = {'欧氏距离', '巴氏距离', '相关系数', '卡方距离'};

% 计算各种距离
for i = 1:length(compare_indices)
    for j = 1:length(compare_indices)
        h1 = histograms{i};
        h2 = histograms{j};
        
        % 欧氏距离
        euclidean_dist = sqrt(sum((h1 - h2).^2));
        
        % 巴氏距离
        bhattacharyya_dist = -log(sum(sqrt(h1 .* h2)));
        
        % 相关系数
        correlation = sum((h1 - mean(h1)) .* (h2 - mean(h2))) / ...
                     (sqrt(sum((h1 - mean(h1)).^2)) * sqrt(sum((h2 - mean(h2)).^2)));
        
        % 卡方距离
        chi_square_dist = sum(((h1 - h2).^2) ./ (h1 + h2 + eps));
        
        similarity_matrix(i, j, :) = [euclidean_dist, bhattacharyya_dist, correlation, chi_square_dist];
    end
end

% 显示相似度矩阵（欧氏距离）
subplot(2, length(compare_indices), [length(compare_indices)+1, length(compare_indices)+2]);
imagesc(similarity_matrix(:, :, 1));
colorbar;
title('直方图欧氏距离相似度矩阵');
xlabel('图像索引');
ylabel('图像索引');
set(gca, 'XTick', 1:length(compare_indices));
set(gca, 'YTick', 1:length(compare_indices));

% 显示具体数值
for i = 1:length(compare_indices)
    for j = 1:length(compare_indices)
        text(j, i, sprintf('%.2f', similarity_matrix(i, j, 1)), ...
             'HorizontalAlignment', 'center', 'Color', 'white');
    end
end

% 显示距离对比
subplot(2, length(compare_indices), [length(compare_indices)+3, length(compare_indices)+4]);
hold on;
colors = lines(4);
for dist_idx = 1:4
    % 计算平均距离
    dist_values = [];
    for i = 1:length(compare_indices)
        for j = i+1:length(compare_indices)
            dist_values = [dist_values, similarity_matrix(i, j, dist_idx)];
        end
    end
    plot(dist_idx, mean(dist_values), 'o', 'Color', colors(dist_idx, :), ...
         'MarkerFaceColor', colors(dist_idx, :), 'MarkerSize', 10);
end
xlim([0 5]);
set(gca, 'XTick', 1:4);
set(gca, 'XTickLabel', methods);
title('不同距离度量平均值');
ylabel('距离值');
grid on;

%% 8. 直方图变换效果对比
fprintf('\n进行直方图变换效果对比...\n');

figure('Name', '直方图变换效果对比', 'Position', [100, 100, 1200, 800]);

% 对数变换
img_log = uint8(255 * log(1 + double(selected_img)) / log(256));

% 幂次变换（gamma校正）
gamma = 2.2;
img_gamma = uint8(255 * (double(selected_img)/255).^(1/gamma));

% 对比度拉伸
img_stretch = imadjust(selected_img);

% 显示各种变换结果
transforms = {'原始图像', '直方图均衡化', '对数变换', 'Gamma校正', '对比度拉伸'};
transformed_imgs = {selected_img, img_eq, img_log, img_gamma, img_stretch};

for i = 1:5
    % 显示图像
    subplot(4, 5, i);
    imshow(transformed_imgs{i});
    title(transforms{i});
    
    % 显示直方图
    subplot(4, 5, i+5);
    [counts, gray_levels] = custom_histogram(transformed_imgs{i});
    bar(gray_levels, counts, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
    xlim([0 255]);
    grid on;
    
    % 计算并显示统计信息
    stats = compute_histogram_stats(double(transformed_imgs{i}(:)));
    
    % 显示累积分布函数
    subplot(4, 5, i+10);
    cdf = cumsum(counts) / sum(counts);
    plot(0:255, cdf, 'b-', 'LineWidth', 1.5);
    xlim([0 255]);
    ylim([0 1]);
    grid on;
    title('累积分布函数');
    
    % 显示统计特征
    subplot(4, 5, i+15);
    axis off;
    text(0.1, 0.9, sprintf('均值: %.1f', stats.mean_value), 'FontSize', 8);
    text(0.1, 0.7, sprintf('标准差: %.1f', stats.std_value), 'FontSize', 8);
    text(0.1, 0.5, sprintf('熵: %.2f', stats.entropy), 'FontSize', 8);
end

%% 9. 直方图在图像增强中的应用
fprintf('\n演示直方图在图像增强中的应用...\n');

% 使用一幅低对比度图像
low_contrast_img = images{2}; % pout图像

figure('Name', '直方图在图像增强中的应用', 'Position', [150, 150, 1000, 600]);

% 原始低对比度图像
subplot(2, 3, 1);
imshow(low_contrast_img);
title('低对比度图像');

subplot(2, 3, 4);
[counts_low, gray_low] = custom_histogram(low_contrast_img);
bar(gray_low, counts_low, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('原始直方图');
xlim([0 255]);
grid on;

% 自适应直方图均衡化（CLAHE）
img_clahe = adapthisteq(low_contrast_img);

subplot(2, 3, 2);
imshow(img_clahe);
title('CLAHE增强');

subplot(2, 3, 5);
[counts_clahe, gray_clahe] = custom_histogram(img_clahe);
bar(gray_clahe, counts_clahe, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('CLAHE直方图');
xlim([0 255]);
grid on;

% 局部直方图均衡化
% 将图像分成4x4块，对每块进行均衡化
blocks = 4;
[m, n] = size(low_contrast_img);
block_h = floor(m / blocks);
block_w = floor(n / blocks);
img_local_eq = low_contrast_img;

for i = 1:blocks
    for j = 1:blocks
        row_start = (i-1)*block_h + 1;
        row_end = min(i*block_h, m);
        col_start = (j-1)*block_w + 1;
        col_end = min(j*block_w, n);
        
        block = low_contrast_img(row_start:row_end, col_start:col_end);
        block_eq = histeq(block);
        img_local_eq(row_start:row_end, col_start:col_end) = block_eq;
    end
end

subplot(2, 3, 3);
imshow(img_local_eq);
title('局部直方图均衡化');

subplot(2, 3, 6);
[counts_local, gray_local] = custom_histogram(img_local_eq);
bar(gray_local, counts_local, 'BarWidth', 1, 'FaceColor', [0.5 0.5 0.5]);
title('局部均衡化直方图');
xlim([0 255]);
grid on;

%% 10. 交互式直方图分析工具
fprintf('\n创建交互式直方图分析工具...\n');

figure('Name', '交互式直方图分析工具', 'Position', [200, 200, 1000, 700]);

% 创建控制面板
uicontrol('Style', 'text', 'Position', [20, 650, 120, 20], ...
          'String', '选择图像:', 'FontSize', 10);
image_popup = uicontrol('Style', 'popupmenu', 'Position', [20, 630, 120, 20], ...
                        'String', img_titles, ...
                        'Callback', @update_histogram_analysis);

uicontrol('Style', 'text', 'Position', [160, 650, 120, 20], ...
          'String', '变换类型:', 'FontSize', 10);
transform_popup = uicontrol('Style', 'popupmenu', 'Position', [160, 630, 120, 20], ...
                            'String', {'无', '均衡化', '对数变换', 'Gamma校正', '对比度拉伸'}, ...
                            'Callback', @update_histogram_analysis);

uicontrol('Style', 'text', 'Position', [300, 650, 120, 20], ...
          'String', 'Gamma值:', 'FontSize', 10);
gamma_slider = uicontrol('Style', 'slider', 'Position', [300, 630, 120, 20], ...
                         'Min', 0.1, 'Max', 5, 'Value', 1, ...
                         'Callback', @update_histogram_analysis);

uicontrol('Style', 'text', 'Position', [440, 650, 120, 20], ...
          'String', '阈值分割:', 'FontSize', 10);
threshold_slider = uicontrol('Style', 'slider', 'Position', [440, 630, 120, 20], ...
                             'Min', 0, 'Max', 255, 'Value', 128, ...
                             'Callback', @update_histogram_analysis);

% 显示图像的坐标轴
ax1 = subplot(2, 3, 1); % 原始图像
ax2 = subplot(2, 3, 2); % 变换后图像
ax3 = subplot(2, 3, 3); % 二值图像
ax4 = subplot(2, 3, 4); % 原始直方图
ax5 = subplot(2, 3, 5); % 变换后直方图
ax6 = subplot(2, 3, 6); % 直方图比较

% 初始显示
update_histogram_analysis();



%% 11. 实验总结
fprintf('\n实验完成！\n');
fprintf('================================================================\n');
fprintf('实验总结:\n');
fprintf('1. 直方图反映了图像中各个灰度级的分布情况\n');
fprintf('2. 直方图均衡化可以增强图像对比度，使灰度分布更均匀\n');
fprintf('3. 直方图匹配可以将一幅图像的直方图转换为另一幅图像的直方图\n');
fprintf('4. 通过直方图统计特征可以分析图像的质量和特性\n');
fprintf('5. 直方图在图像分割、增强和匹配中有重要应用\n');
fprintf('================================================================\n');

%% ====================================================
%% 所有函数定义（已移动到文件末尾）
%% ====================================================

%% 1. 自定义直方图计算函数
function [counts, gray_levels] = custom_histogram(img)
    % 计算图像的直方图
    % 输入参数：
    %   img - 输入图像（uint8类型）
    % 输出参数：
    %   counts - 每个灰度级的像素数
    %   gray_levels - 灰度级（0-255）
    
    % 确保图像是uint8类型
    if ~isa(img, 'uint8')
        img = uint8(img);
    end
    
    % 初始化直方图
    counts = zeros(256, 1);
    
    % 计算直方图
    for i = 0:255
        counts(i+1) = sum(img(:) == i);
    end
    
    gray_levels = 0:255;
end

%% 2. 自定义直方图均衡化函数
function img_eq = custom_histeq(img)
    % 直方图均衡化
    % 输入参数：
    %   img - 输入图像（uint8类型）
    % 输出参数：
    %   img_eq - 均衡化后的图像
    
    % 计算直方图
    [counts, ~] = custom_histogram(img);
    
    % 计算累积分布函数（CDF）
    cdf = cumsum(counts);
    
    % 归一化CDF
    cdf_normalized = cdf / cdf(end);
    
    % 映射到新的灰度级
    img_eq = uint8(255 * cdf_normalized(double(img) + 1));
end

%% 3. 自定义直方图匹配函数
function img_matched = custom_histmatch(source_img, target_img)
    % 直方图匹配（规定化）
    % 输入参数：
    %   source_img - 源图像
    %   target_img - 目标图像
    % 输出参数：
    %   img_matched - 匹配后的图像
    
    % 计算源图像和目标图像的直方图
    [counts_source, ~] = custom_histogram(source_img);
    [counts_target, ~] = custom_histogram(target_img);
    
    % 计算累积分布函数
    cdf_source = cumsum(counts_source) / sum(counts_source);
    cdf_target = cumsum(counts_target) / sum(counts_target);
    
    % 创建映射表
    mapping = zeros(256, 1);
    
    for i = 1:256
        % 找到目标CDF中最接近源CDF的值
        [~, j] = min(abs(cdf_target - cdf_source(i)));
        mapping(i) = j - 1; % 转换为0-255范围
    end
    
    % 应用映射
    img_matched = uint8(mapping(double(source_img) + 1));
end

%% 4. 直方图统计特征计算函数
function stats = compute_histogram_stats(img_data)
    % 计算直方图统计特征
    % 输入参数：
    %   img_data - 图像数据（向量形式）
    % 输出参数：
    %   stats - 包含各种统计特征的结构体
    
    stats = struct();
    
    % 基本统计量
    stats.total_pixels = length(img_data);
    stats.mean_value = mean(img_data);
    stats.median_value = median(img_data);
    stats.std_value = std(img_data);
    stats.variance = var(img_data);
    
    % 高阶矩（偏度和峰度）
    stats.skewness = skewness(img_data);
    stats.kurtosis = kurtosis(img_data);
    
    % 最小值、最大值、范围
    stats.min_value = min(img_data);
    stats.max_value = max(img_data);
    stats.range = stats.max_value - stats.min_value;
    
    % 熵
    [counts, ~] = histcounts(img_data, 256);
    prob = counts / sum(counts);
    prob(prob == 0) = []; % 移除零概率
    stats.entropy = -sum(prob .* log2(prob));
end

%% 5. 大津法（Otsu）阈值分割函数
function [threshold, binary_img] = otsu_threshold(img)
    % 大津法（Otsu）阈值分割
    % 输入参数：
    %   img - 输入图像
    % 输出参数：
    %   threshold - 计算得到的最佳阈值
    %   binary_img - 二值分割结果
    
    % 计算直方图
    [counts, ~] = custom_histogram(img);
    total_pixels = numel(img);
    
    % 归一化直方图
    p = counts / total_pixels;
    
    % 初始化变量
    max_variance = 0;
    threshold = 0;
    
    % 遍历所有可能的阈值
    for t = 0:254
        % 计算两类概率
        w0 = sum(p(1:t+1));
        w1 = sum(p(t+2:end));
        
        if w0 == 0 || w1 == 0
            continue;
        end
        
        % 计算两类均值
        mu0 = sum((0:t)' .* p(1:t+1)) / w0;
        mu1 = sum((t+1:255)' .* p(t+2:end)) / w1;
        
        % 计算类间方差
        variance = w0 * w1 * (mu0 - mu1)^2;
        
        % 更新最大方差和阈值
        if variance > max_variance
            max_variance = variance;
            threshold = t;
        end
    end
    
    % 应用阈值
    binary_img = img > threshold;
end

% 更新函数
    function update_histogram_analysis(~, ~)
        % 获取控制参数
        img_idx = get(image_popup, 'Value');
        transform_idx = get(transform_popup, 'Value');
        gamma_val = get(gamma_slider, 'Value');
        threshold_val = get(threshold_slider, 'Value');
        
        % 获取当前图像
        current_img = images{img_idx};
        
        % 应用变换
        transform_types = {'无', '均衡化', '对数变换', 'Gamma校正', '对比度拉伸'};
        transform_name = transform_types{transform_idx};
        
        switch transform_idx
            case 1 % 无变换
                transformed_img = current_img;
            case 2 % 均衡化
                transformed_img = histeq(current_img);
            case 3 % 对数变换
                transformed_img = uint8(255 * log(1 + double(current_img)) / log(256));
            case 4 % Gamma校正
                transformed_img = uint8(255 * (double(current_img)/255).^(1/gamma_val));
            case 5 % 对比度拉伸
                transformed_img = imadjust(current_img);
        end
        
        % 阈值分割
        binary_img = current_img > threshold_val;
        
        % 显示原始图像
        axes(ax1);
        imshow(current_img);
        title(sprintf('原始图像\n%s', img_titles{img_idx}));
        
        % 显示变换后图像
        axes(ax2);
        imshow(transformed_img);
        title(sprintf('变换后图像\n%s', transform_name));
        
        % 显示二值图像
        axes(ax3);
        imshow(binary_img);
        title(sprintf('二值分割\n阈值 = %d', threshold_val));
        
        % 计算并显示直方图
        [counts_orig, gray_orig] = custom_histogram(current_img);
        [counts_trans, gray_trans] = custom_histogram(transformed_img);
        
        % 显示原始直方图
        axes(ax4);
        bar(gray_orig, counts_orig, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
        hold on;
        line([threshold_val threshold_val], [0 max(counts_orig)], ...
             'Color', 'r', 'LineWidth', 2, 'LineStyle', '--');
        hold off;
        xlim([0 255]);
        title('原始直方图');
        xlabel('灰度级');
        ylabel('像素数');
        grid on;
        
        % 显示变换后直方图
        axes(ax5);
        bar(gray_trans, counts_trans, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
        xlim([0 255]);
        title('变换后直方图');
        xlabel('灰度级');
        ylabel('像素数');
        grid on;
        
        % 显示直方图比较
        axes(ax6);
        hold off;
        plot(0:255, counts_orig/sum(counts_orig), 'b-', 'LineWidth', 2, 'DisplayName', '原始直方图');
        hold on;
        plot(0:255, counts_trans/sum(counts_trans), 'r-', 'LineWidth', 2, 'DisplayName', '变换后直方图');
        xlim([0 255]);
        title('直方图对比（归一化）');
        xlabel('灰度级');
        ylabel('归一化频率');
        legend('Location', 'best');
        grid on;
        
        % 计算统计特征
        stats_orig = compute_histogram_stats(double(current_img(:)));
        stats_trans = compute_histogram_stats(double(transformed_img(:)));
        
        % 在图像上显示统计信息
        axes(ax2);
        text(10, 20, sprintf('均值: %.1f', stats_trans.mean_value), ...
             'Color', 'white', 'FontSize', 9, 'BackgroundColor', 'black');
        text(10, 40, sprintf('标准差: %.1f', stats_trans.std_value), ...
             'Color', 'white', 'FontSize', 9, 'BackgroundColor', 'black');
    end