%% 实验五：直方图均匀化实验
clear; close all; clc;
fprintf('=== 直方图均匀化（均衡化）实验 ===\n\n');

%% 1. 实验原理说明
fprintf('实验原理：\n');
fprintf('直方图均衡化是一种通过改变图像直方图来增强图像对比度的技术。\n');
fprintf('它将原始图像的直方图转换为近似均匀分布，从而展宽像素值分布范围。\n\n');

%% 2. 加载测试图像
fprintf('加载测试图像...\n');

% 创建低对比度图像作为示例
% 示例1：曝光不足的图像（暗图像）
img_dark = imread('pout.tif'); % 这是一个典型的低对比度图像
if size(img_dark, 3) == 3
    img_dark = rgb2gray(img_dark);
end

% 示例2：高对比度但分布不均的图像
img_bright = imread('cameraman.tif');
if size(img_bright, 3) == 3
    img_bright = rgb2gray(img_bright);
end

% 示例3：创建人工低对比度图像
img_low_contrast = uint8(50 + randn(256, 256) * 20); % 均值50，标准差20
img_low_contrast = max(0, min(255, img_low_contrast));

% 示例4：创建双峰直方图图像（适合展示均衡化效果）
[m, n] = size(img_bright);
img_bimodal = zeros(m, n, 'uint8');
for i = 1:m
    for j = 1:n
        if rand() > 0.5
            img_bimodal(i, j) = uint8(50 + randn() * 10);
        else
            img_bimodal(i, j) = uint8(200 + randn() * 10);
        end
    end
end

% 显示原始图像
figure('Name', '原始测试图像', 'Position', [100, 100, 1200, 800]);

% 暗图像
subplot(2, 4, 1);
imshow(img_dark);
title('曝光不足图像 (pout.tif)');
subplot(2, 4, 5);
imhist(img_dark);
title('直方图');
xlim([0 255]);

% 亮图像
subplot(2, 4, 2);
imshow(img_bright);
title('正常图像 (cameraman.tif)');
subplot(2, 4, 6);
imhist(img_bright);
title('直方图');
xlim([0 255]);

% 低对比度图像
subplot(2, 4, 3);
imshow(img_low_contrast);
title('人工低对比度图像');
subplot(2, 4, 7);
imhist(img_low_contrast);
title('直方图');
xlim([0 255]);

% 双峰图像
subplot(2, 4, 4);
imshow(img_bimodal);
title('双峰直方图图像');
subplot(2, 4, 8);
imhist(img_bimodal);
title('直方图');
xlim([0 255]);

%% 3. 直方图均衡化步骤演示（详细步骤）
fprintf('\n演示直方图均衡化的详细步骤...\n');

% 选择一幅图像进行详细步骤演示
demo_img = img_dark; % 使用曝光不足的图像

figure('Name', '直方图均衡化详细步骤', 'Position', [150, 150, 1400, 800]);

% 步骤1：原始图像和直方图
subplot(3, 4, 1);
imshow(demo_img);
title('原始图像');

subplot(3, 4, 5);
[counts_orig, gray_levels] = custom_histogram(demo_img);
bar(gray_levels, counts_orig, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
title('原始直方图');
xlabel('灰度级');
ylabel('像素数');
xlim([0 255]);
grid on;

% 计算原始直方图的统计信息
stats_orig = compute_image_stats(demo_img);
text(100, max(counts_orig)*0.9, sprintf('均值: %.1f', stats_orig.mean), 'FontSize', 8);
text(100, max(counts_orig)*0.8, sprintf('标准差: %.1f', stats_orig.std), 'FontSize', 8);

% 步骤2：计算概率分布函数（PDF）
subplot(3, 4, 2);
pdf_orig = counts_orig / sum(counts_orig);
bar(gray_levels, pdf_orig, 'BarWidth', 1, 'FaceColor', [0.7 0.7 0.9]);
title('概率分布函数 (PDF)');
xlabel('灰度级');
ylabel('概率');
xlim([0 255]);
grid on;

% 步骤3：计算累积分布函数（CDF）
subplot(3, 4, 6);
cdf_orig = cumsum(pdf_orig);
bar(gray_levels, cdf_orig, 'BarWidth', 1, 'FaceColor', [0.9 0.7 0.7]);
title('累积分布函数 (CDF)');
xlabel('灰度级');
ylabel('累积概率');
xlim([0 255]);
ylim([0 1]);
grid on;

% 步骤4：计算均衡化映射函数
subplot(3, 4, 3);
% 映射函数：s_k = T(r_k) = round(255 * CDF(r_k))
mapping_func = round(255 * cdf_orig);
plot(0:255, mapping_func, 'b-', 'LineWidth', 2);
hold on;
plot(0:255, 0:255, 'r--', 'LineWidth', 1); % 对角线 y=x
title('均衡化映射函数');
xlabel('原始灰度级 r_k');
ylabel('新灰度级 s_k');
xlim([0 255]);
ylim([0 255]);
grid on;
legend('映射函数', 'y = x', 'Location', 'northwest');

% 步骤5：应用映射函数得到均衡化图像
% 使用自定义的直方图均衡化函数
img_eq = histogram_equalization(demo_img);

subplot(3, 4, 4);
imshow(img_eq);
title('均衡化后的图像');

% 步骤6：均衡化后的直方图
subplot(3, 4, 8);
[counts_eq, gray_eq] = custom_histogram(img_eq);
bar(gray_eq, counts_eq, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
title('均衡化后的直方图');
xlabel('灰度级');
ylabel('像素数');
xlim([0 255]);
grid on;

% 计算均衡化后的统计信息
stats_eq = compute_image_stats(img_eq);
text(100, max(counts_eq)*0.9, sprintf('均值: %.1f', stats_eq.mean), 'FontSize', 8);
text(100, max(counts_eq)*0.8, sprintf('标准差: %.1f', stats_eq.std), 'FontSize', 8);

% 步骤7：均衡化后的PDF
subplot(3, 4, 7);
pdf_eq = counts_eq / sum(counts_eq);
bar(gray_eq, pdf_eq, 'BarWidth', 1, 'FaceColor', [0.7 0.7 0.9]);
title('均衡化后的PDF');
xlabel('灰度级');
ylabel('概率');
xlim([0 255]);
grid on;

% 步骤8：均衡化后的CDF
subplot(3, 4, [11, 12]);
cdf_eq = cumsum(pdf_eq);
bar(gray_eq, cdf_eq, 'BarWidth', 1, 'FaceColor', [0.9 0.7 0.7]);
hold on;
plot(0:255, (0:255)/255, 'k--', 'LineWidth', 1); % 理想均匀分布的CDF
title('均衡化后的CDF (对比理想均匀分布)');
xlabel('灰度级');
ylabel('累积概率');
xlim([0 255]);
ylim([0 1]);
grid on;
legend('均衡化CDF', '理想均匀分布', 'Location', 'northwest');

%% 4. 不同图像的均衡化效果对比
fprintf('\n比较不同图像的均衡化效果...\n');

test_images = {img_dark, img_bright, img_low_contrast, img_bimodal};
test_names = {'曝光不足图像', '正常图像', '低对比度图像', '双峰图像'};

figure('Name', '不同图像的均衡化效果对比', 'Position', [100, 100, 1400, 1000]);

for i = 1:length(test_images)
    img = test_images{i};
    img_eq = histogram_equalization(img);
    
    % 计算均衡化前后的统计信息
    stats_orig = compute_image_stats(img);
    stats_eq = compute_image_stats(img_eq);
    
    % 显示原始图像
    subplot(4, 4, (i-1)*4 + 1);
    imshow(img);
    title(test_names{i});
    
    % 显示原始直方图
    subplot(4, 4, (i-1)*4 + 2);
    [counts_orig, gray_orig] = custom_histogram(img);
    bar(gray_orig, counts_orig, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
    xlim([0 255]);
    title('原始直方图');
    grid on;
    
    % 显示均衡化后的图像
    subplot(4, 4, (i-1)*4 + 3);
    imshow(img_eq);
    title('均衡化后');
    
    % 显示均衡化后的直方图
    subplot(4, 4, (i-1)*4 + 4);
    [counts_eq, gray_eq] = custom_histogram(img_eq);
    bar(gray_eq, counts_eq, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
    xlim([0 255]);
    title('均衡化直方图');
    grid on;
    
    % 在图像上添加统计信息
    subplot(4, 4, (i-1)*4 + 1);
    text(10, 20, sprintf('均值: %.1f', stats_orig.mean), ...
         'Color', 'white', 'FontSize', 8, 'BackgroundColor', 'black');
    text(10, 40, sprintf('标准差: %.1f', stats_orig.std), ...
         'Color', 'white', 'FontSize', 8, 'BackgroundColor', 'black');
    
    subplot(4, 4, (i-1)*4 + 3);
    text(10, 20, sprintf('均值: %.1f', stats_eq.mean), ...
         'Color', 'white', 'FontSize', 8, 'BackgroundColor', 'black');
    text(10, 40, sprintf('标准差: %.1f', stats_eq.std), ...
         'Color', 'white', 'FontSize', 8, 'BackgroundColor', 'black');
end

%% 5. 均衡化映射函数分析
fprintf('\n分析均衡化映射函数...\n');

figure('Name', '均衡化映射函数分析', 'Position', [150, 150, 1200, 800]);

% 计算不同图像的映射函数
mapping_functions = cell(1, length(test_images));

for i = 1:length(test_images)
    img = test_images{i};
    [counts, ~] = custom_histogram(img);
    pdf = counts / sum(counts);
    cdf = cumsum(pdf);
    mapping_functions{i} = round(255 * cdf);
end

% 绘制所有映射函数
subplot(2, 3, 1);
hold on;
colors = lines(length(test_images));
for i = 1:length(test_images)
    plot(0:255, mapping_functions{i}, 'Color', colors(i, :), 'LineWidth', 2, ...
         'DisplayName', test_names{i});
end
plot(0:255, 0:255, 'k--', 'LineWidth', 1, 'DisplayName', 'y = x');
title('不同图像的均衡化映射函数');
xlabel('原始灰度级 r_k');
ylabel('新灰度级 s_k');
xlim([0 255]);
ylim([0 255]);
legend('Location', 'northwest');
grid on;

% 分析映射函数的斜率
subplot(2, 3, 2);
hold on;
for i = 1:length(test_images)
    % 计算映射函数的导数（近似）
    mapping = mapping_functions{i};
    derivative = diff(mapping);
    plot(0:254, derivative, 'Color', colors(i, :), 'LineWidth', 1.5, ...
         'DisplayName', test_names{i});
end
title('映射函数的导数（斜率）');
xlabel('原始灰度级 r_k');
ylabel('斜率');
xlim([0 255]);
grid on;
legend('Location', 'best');

% 分析像素值变化
subplot(2, 3, 3);
hold on;
for i = 1:length(test_images)
    img = test_images{i};
    img_eq = histogram_equalization(img);
    
    % 计算像素值变化
    diff_img = double(img_eq) - double(img);
    hist_diff = histogram(diff_img(:), -255:10:255, 'Normalization', 'probability');
    hist_diff.FaceColor = colors(i, :);
    hist_diff.FaceAlpha = 0.5;
end
title('像素值变化分布');
xlabel('像素值变化量');
ylabel('概率');
grid on;
legend(test_names, 'Location', 'best');

% 分析直方图形状变化
subplot(2, 3, 4);
hold on;
for i = 1:length(test_images)
    img = test_images{i};
    [counts_orig, ~] = custom_histogram(img);
    pdf_orig = counts_orig / sum(counts_orig);
    
    % 计算直方图平坦度（熵）
    entropy_orig = -sum(pdf_orig(pdf_orig > 0) .* log2(pdf_orig(pdf_orig > 0)));
    
    img_eq = histogram_equalization(img);
    [counts_eq, ~] = custom_histogram(img_eq);
    pdf_eq = counts_eq / sum(counts_eq);
    entropy_eq = -sum(pdf_eq(pdf_eq > 0) .* log2(pdf_eq(pdf_eq > 0)));
    
    plot(i, entropy_orig, 'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
         'MarkerSize', 10, 'DisplayName', test_names{i});
    plot(i, entropy_eq, 's', 'Color', colors(i, :), 'MarkerFaceColor', 'none', ...
         'MarkerSize', 10, 'LineWidth', 2);
end
title('均衡化前后熵值对比');
xlabel('图像类型');
ylabel('熵值');
xlim([0 length(test_images)+1]);
set(gca, 'XTick', 1:length(test_images));
set(gca, 'XTickLabel', test_names);
grid on;
legend({'原始熵值', '均衡化后熵值'}, 'Location', 'best');

% 分析对比度增强效果
subplot(2, 3, [5, 6]);
hold on;
for i = 1:length(test_images)
    img = test_images{i};
    stats_orig = compute_image_stats(img);
    
    img_eq = histogram_equalization(img);
    stats_eq = compute_image_stats(img_eq);
    
    % 绘制对比度增强指标
    bar_locs = (i-1)*3+1:(i-1)*3+2;
    bar_heights = [stats_orig.std, stats_eq.std];
    
    bar(bar_locs, bar_heights, 'FaceColor', colors(i, :));
    text(mean(bar_locs), max(bar_heights)*1.05, test_names{i}, ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end
title('均衡化前后标准差对比（对比度指标）');
ylabel('标准差');
xlabel('图像类型');
set(gca, 'XTick', []);
grid on;
legend({'原始标准差', '均衡化后标准差'}, 'Location', 'best');

%% 6. 对比不同均衡化方法
fprintf('\n对比不同均衡化方法...\n');

% 测试图像
test_img = img_dark;

% 方法1：自定义直方图均衡化
img_custom_eq = histogram_equalization(test_img);

% 方法2：MATLAB内置histeq函数
img_matlabeq = histeq(test_img);

% 方法3：自适应直方图均衡化（AHE）
img_ahe = adapthisteq(test_img);

% 方法4：对比度受限的自适应直方图均衡化（CLAHE）
img_clahe = adapthisteq(test_img, 'ClipLimit', 0.02, 'Distribution', 'uniform');

figure('Name', '不同均衡化方法对比', 'Position', [100, 100, 1400, 1000]);

methods = {'原始图像', '自定义均衡化', 'MATLAB histeq', '自适应均衡化(AHE)', 'CLAHE'};
images = {test_img, img_custom_eq, img_matlabeq, img_ahe, img_clahe};

for i = 1:length(methods)
    % 显示图像
    subplot(3, 5, i);
    imshow(images{i});
    title(methods{i});
    
    % 显示直方图
    subplot(3, 5, i+5);
    [counts, gray_levels] = custom_histogram(images{i});
    bar(gray_levels, counts, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
    xlim([0 255]);
    title('直方图');
    grid on;
    
    % 计算并显示统计信息
    stats = compute_image_stats(images{i});
    
    % 显示CDF
    subplot(3, 5, i+10);
    pdf = counts / sum(counts);
    cdf = cumsum(pdf);
    bar(gray_levels, cdf, 'BarWidth', 1, 'FaceColor', [0.8 0.6 0.6]);
    hold on;
    plot(0:255, (0:255)/255, 'k--', 'LineWidth', 1); % 理想均匀分布
    xlim([0 255]);
    ylim([0 1]);
    title('CDF');
    grid on;
    
    % 在图像上添加统计信息
    subplot(3, 5, i);
    text(10, 20, sprintf('均值: %.1f', stats.mean), ...
         'Color', 'white', 'FontSize', 8, 'BackgroundColor', 'black');
    text(10, 40, sprintf('标准差: %.1f', stats.std), ...
         'Color', 'white', 'FontSize', 8, 'BackgroundColor', 'black');
end

%% 7. 均衡化对图像质量的影响分析
fprintf('\n分析均衡化对图像质量的影响...\n');

% 创建测试图像：添加噪声后均衡化
noisy_img = imnoise(test_img, 'gaussian', 0, 0.01); % 添加高斯噪声
noisy_img_eq = histogram_equalization(noisy_img);

% 创建测试图像：添加纹理后均衡化
texture_img = test_img;
[m, n] = size(texture_img);
[X, Y] = meshgrid(1:n, 1:m);
texture = uint8(50 * sin(X/20) .* sin(Y/20)); % 正弦波纹理
texture_img = uint8(min(255, max(0, double(test_img) + double(texture))));
texture_img_eq = histogram_equalization(texture_img);

figure('Name', '均衡化对图像质量的影响', 'Position', [150, 150, 1200, 800]);

% 噪声图像分析
subplot(2, 4, 1);
imshow(noisy_img);
title('带噪声的图像');

subplot(2, 4, 2);
imshow(noisy_img_eq);
title('均衡化后的噪声图像');

subplot(2, 4, 5);
[counts_noisy, ~] = custom_histogram(noisy_img);
bar(0:255, counts_noisy, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
xlim([0 255]);
title('噪声图像直方图');
grid on;

subplot(2, 4, 6);
[counts_noisy_eq, ~] = custom_histogram(noisy_img_eq);
bar(0:255, counts_noisy_eq, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
xlim([0 255]);
title('均衡化后直方图');
grid on;

% 纹理图像分析
subplot(2, 4, 3);
imshow(texture_img);
title('带纹理的图像');

subplot(2, 4, 4);
imshow(texture_img_eq);
title('均衡化后的纹理图像');

subplot(2, 4, 7);
[counts_texture, ~] = custom_histogram(texture_img);
bar(0:255, counts_texture, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
xlim([0 255]);
title('纹理图像直方图');
grid on;

subplot(2, 4, 8);
[counts_texture_eq, ~] = custom_histogram(texture_img_eq);
bar(0:255, counts_texture_eq, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
xlim([0 255]);
title('均衡化后直方图');
grid on;

%% 8. 交互式直方图均衡化工具 - 修正版本
fprintf('\n创建交互式直方图均衡化工具...\n');

% 创建交互式图形窗口
hFig = figure('Name', '交互式直方图均衡化工具', 'Position', [200, 200, 1000, 700]);

% 图像数据
image_types = {'曝光不足', '正常图像', '低对比度', '双峰图像', '带噪声', '带纹理'};
image_data = {img_dark, img_bright, img_low_contrast, img_bimodal, noisy_img, texture_img};

% 创建控制面板
uicontrol('Style', 'text', 'Position', [20, 650, 120, 20], ...
          'String', '选择图像类型:', 'FontSize', 10);
image_popup = uicontrol('Style', 'popupmenu', 'Position', [20, 630, 120, 20], ...
                        'String', image_types, ...
                        'Callback', @update_equalization);

uicontrol('Style', 'text', 'Position', [160, 650, 120, 20], ...
          'String', '均衡化方法:', 'FontSize', 10);
method_popup = uicontrol('Style', 'popupmenu', 'Position', [160, 630, 120, 20], ...
                         'String', {'自定义均衡化', 'MATLAB histeq', 'AHE', 'CLAHE'}, ...
                         'Callback', @update_equalization);

uicontrol('Style', 'text', 'Position', [300, 650, 120, 20], ...
          'String', '显示映射函数:', 'FontSize', 10);
show_mapping_checkbox = uicontrol('Style', 'checkbox', 'Position', [300, 630, 120, 20], ...
                                  'String', '是', 'Value', 1, ...
                                  'Callback', @update_equalization);

uicontrol('Style', 'text', 'Position', [440, 650, 120, 20], ...
          'String', '显示统计信息:', 'FontSize', 10);
show_stats_checkbox = uicontrol('Style', 'checkbox', 'Position', [440, 630, 120, 20], ...
                                'String', '是', 'Value', 1, ...
                                'Callback', @update_equalization);

% 存储数据到图形对象
setappdata(hFig, 'image_data', image_data);
setappdata(hFig, 'image_types', image_types);

% 显示图像的坐标轴
ax1 = subplot(2, 3, 1); % 原始图像
ax2 = subplot(2, 3, 2); % 均衡化后图像
ax3 = subplot(2, 3, 3); % 映射函数
ax4 = subplot(2, 3, 4); % 原始直方图
ax5 = subplot(2, 3, 5); % 均衡化后直方图
ax6 = subplot(2, 3, 6); % 直方图对比

% 初始显示
update_equalization();



%% 9. 实验总结与理论验证
fprintf('\n实验总结与理论验证...\n');

% 验证直方图均衡化的数学原理
figure('Name', '直方图均衡化理论验证', 'Position', [100, 100, 1200, 600]);

% 验证1：CDF变换
subplot(1, 3, 1);
img = img_dark;
[counts, ~] = custom_histogram(img);
pdf = counts / sum(counts);
cdf_original = cumsum(pdf);

img_eq = histogram_equalization(img);
[counts_eq, ~] = custom_histogram(img_eq);
pdf_eq = counts_eq / sum(counts_eq);
cdf_equalized = cumsum(pdf_eq);

plot(0:255, cdf_original, 'b-', 'LineWidth', 2, 'DisplayName', '原始CDF');
hold on;
plot(0:255, cdf_equalized, 'r-', 'LineWidth', 2, 'DisplayName', '均衡化后CDF');
plot(0:255, (0:255)/255, 'k--', 'LineWidth', 1, 'DisplayName', '理想均匀CDF');
title('CDF变换验证');
xlabel('灰度级');
ylabel('累积概率');
legend('Location', 'best');
grid on;

% 验证2：直方图平坦度
subplot(1, 3, 2);
% 计算直方图的均匀性指标
uniformity_orig = sum((pdf - 1/256).^2); % 与均匀分布的差异
uniformity_eq = sum((pdf_eq - 1/256).^2);

bar([1, 2], [uniformity_orig, uniformity_eq], 'FaceColor', [0.6 0.6 0.8]);
set(gca, 'XTickLabel', {'原始直方图', '均衡化直方图'});
ylabel('与均匀分布的差异');
title('直方图平坦度验证');
grid on;

% 验证3：信息熵变化
subplot(1, 3, 3);
% 计算信息熵
entropy_orig = -sum(pdf(pdf > 0) .* log2(pdf(pdf > 0)));
entropy_eq = -sum(pdf_eq(pdf_eq > 0) .* log2(pdf_eq(pdf_eq > 0)));

bar([1, 2], [entropy_orig, entropy_eq], 'FaceColor', [0.8 0.6 0.6]);
set(gca, 'XTickLabel', {'原始图像', '均衡化图像'});
ylabel('信息熵 (bits)');
title('信息熵变化验证');
ylim([0 max([entropy_orig, entropy_eq]) * 1.2]);
grid on;

% 添加数值标签
text(1, entropy_orig, sprintf('%.3f', entropy_orig), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
text(2, entropy_eq, sprintf('%.3f', entropy_eq), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);

%% 10. 实验结论
fprintf('\n实验完成！\n');
fprintf('================================================================\n');
fprintf('实验结论：\n');
fprintf('1. 直方图均衡化通过将原始直方图转换为近似均匀分布来增强图像对比度。\n');
fprintf('2. 对于低对比度图像，均衡化效果显著，能展宽灰度级分布。\n');
fprintf('3. 均衡化会使图像的均值趋向于中间灰度级（~128），标准差增大。\n');
fprintf('4. 均衡化可能放大噪声，特别是对于原本噪声较大的图像。\n');
fprintf('5. 自适应均衡化方法（AHE、CLAHE）能更好地处理局部对比度。\n');
fprintf('6. 均衡化映射函数反映了从原始灰度级到新灰度级的转换关系。\n');
fprintf('================================================================\n');

%% ====================================================
%% 所有函数定义（已移动到文件末尾）
%% ====================================================

%%
% 更新函数 - 现在它是图形窗口的本地函数
function update_equalization(~, ~)
    % 从当前图形获取句柄
    hFig = gcf;
    image_data = getappdata(hFig, 'image_data');
    image_types = getappdata(hFig, 'image_types');
    
    % 查找控件
    image_popup = findobj(hFig, 'Style', 'popupmenu', 'Position', [20, 630, 120, 20]);
    method_popup = findobj(hFig, 'Style', 'popupmenu', 'Position', [160, 630, 120, 20]);
    show_mapping_checkbox = findobj(hFig, 'Style', 'checkbox', 'Position', [300, 630, 120, 20]);
    show_stats_checkbox = findobj(hFig, 'Style', 'checkbox', 'Position', [440, 630, 120, 20]);
    
    % 获取控制参数
    img_idx = get(image_popup, 'Value');
    method_idx = get(method_popup, 'Value');
    show_mapping = get(show_mapping_checkbox, 'Value');
    show_stats = get(show_stats_checkbox, 'Value');
    
    % 获取当前图像
    current_img = image_data{img_idx};
    
    % 应用均衡化方法
    method_names = {'自定义均衡化', 'MATLAB histeq', 'AHE', 'CLAHE'};
    
    switch method_idx
        case 1 % 自定义均衡化
            eq_img = histogram_equalization(current_img);
        case 2 % MATLAB histeq
            eq_img = histeq(current_img);
        case 3 % AHE
            eq_img = adapthisteq(current_img);
        case 4 % CLAHE
            eq_img = adapthisteq(current_img, 'ClipLimit', 0.02, 'Distribution', 'uniform');
    end
    
    % 显示原始图像
    axes(ax1);
    imshow(current_img);
    title(sprintf('原始图像\n%s', image_types{img_idx}));
    
    % 显示均衡化后图像
    axes(ax2);
    imshow(eq_img);
    title(sprintf('均衡化后\n%s', method_names{method_idx}));
    
    % 计算并显示映射函数
    if show_mapping
        axes(ax3);
        [counts, ~] = custom_histogram(current_img);
        pdf = counts / sum(counts);
        cdf = cumsum(pdf);
        mapping_func = round(255 * cdf);
        
        plot(0:255, mapping_func, 'b-', 'LineWidth', 2);
        hold on;
        plot(0:255, 0:255, 'r--', 'LineWidth', 1);
        hold off;
        title('均衡化映射函数');
        xlabel('原始灰度级 r_k');
        ylabel('新灰度级 s_k');
        xlim([0 255]);
        ylim([0 255]);
        grid on;
        legend('映射函数', 'y = x', 'Location', 'northwest');
    else
        axes(ax3);
        cla;
        text(0.5, 0.5, '映射函数已隐藏', ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        axis off;
    end
    
    % 计算直方图
    [counts_orig, gray_orig] = custom_histogram(current_img);
    [counts_eq, gray_eq] = custom_histogram(eq_img);
    
    % 显示原始直方图
    axes(ax4);
    bar(gray_orig, counts_orig, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
    xlim([0 255]);
    title('原始直方图');
    xlabel('灰度级');
    ylabel('像素数');
    grid on;
    
    % 显示均衡化后直方图
    axes(ax5);
    bar(gray_eq, counts_eq, 'BarWidth', 1, 'FaceColor', [0.6 0.6 0.6]);
    xlim([0 255]);
    title('均衡化后直方图');
    xlabel('灰度级');
    ylabel('像素数');
    grid on;
    
    % 显示直方图对比
    axes(ax6);
    hold off;
    % 归一化直方图以便比较
    pdf_orig = counts_orig / sum(counts_orig);
    pdf_eq = counts_eq / sum(counts_eq);
    
    bar(gray_orig, pdf_orig, 'BarWidth', 1, 'FaceColor', [0.7 0.7 0.9], 'FaceAlpha', 0.5);
    hold on;
    bar(gray_eq, pdf_eq, 'BarWidth', 1, 'FaceColor', [0.9 0.7 0.7], 'FaceAlpha', 0.5);
    xlim([0 255]);
    title('直方图对比（归一化）');
    xlabel('灰度级');
    ylabel('概率');
    legend('原始直方图', '均衡化直方图', 'Location', 'best');
    grid on;
    
    % 显示统计信息
    if show_stats
        stats_orig = compute_image_stats(current_img);
        stats_eq = compute_image_stats(eq_img);
        
        % 在图像上显示统计信息
        axes(ax1);
        text(10, 20, sprintf('均值: %.1f', stats_orig.mean), ...
             'Color', 'white', 'FontSize', 9, 'BackgroundColor', 'black');
        text(10, 40, sprintf('标准差: %.1f', stats_orig.std), ...
             'Color', 'white', 'FontSize', 9, 'BackgroundColor', 'black');
        
        axes(ax2);
        text(10, 20, sprintf('均值: %.1f', stats_eq.mean), ...
             'Color', 'white', 'FontSize', 9, 'BackgroundColor', 'black');
        text(10, 40, sprintf('标准差: %.1f', stats_eq.std), ...
             'Color', 'white', 'FontSize', 9, 'BackgroundColor', 'black');
    end
end
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

%% 2. 图像统计信息计算函数
function stats = compute_image_stats(img)
    % 计算图像的统计信息
    % 输入参数：
    %   img - 输入图像
    % 输出参数：
    %   stats - 包含各种统计特征的结构体
    
    img_double = double(img(:));
    
    stats.mean = mean(img_double);
    stats.std = std(img_double);
    stats.median = median(img_double);
    stats.min = min(img_double);
    stats.max = max(img_double);
    stats.range = stats.max - stats.min;
    
    % 计算偏度和峰度
    stats.skewness = skewness(img_double);
    stats.kurtosis = kurtosis(img_double);
    
    % 计算熵
    [counts, ~] = histcounts(img_double, 256);
    prob = counts / sum(counts);
    prob(prob == 0) = [];
    stats.entropy = -sum(prob .* log2(prob));
end

%% 3. 直方图均衡化函数
function img_eq = histogram_equalization(img)
    % 直方图均衡化
    % 输入参数：
    %   img - 输入图像（uint8类型）
    % 输出参数：
    %   img_eq - 均衡化后的图像
    
    % 确保输入是uint8类型
    if ~isa(img, 'uint8')
        img = uint8(img);
    end
    
    % 步骤1：计算直方图
    [counts, ~] = custom_histogram(img);
    
    % 步骤2：计算概率分布函数（PDF）
    total_pixels = numel(img);
    pdf = counts / total_pixels;
    
    % 步骤3：计算累积分布函数（CDF）
    cdf = cumsum(pdf);
    
    % 步骤4：计算映射函数
    % s_k = T(r_k) = round(255 * CDF(r_k))
    mapping = round(255 * cdf);
    
    % 步骤5：应用映射函数
    img_eq = uint8(mapping(double(img) + 1)); % +1是因为MATLAB索引从1开始
end

%% 4. 对比度增强指标计算函数
function metrics = compute_contrast_metrics(img_orig, img_eq)
    % 计算对比度增强的指标
    % 输入参数：
    %   img_orig - 原始图像
    %   img_eq - 均衡化后的图像
    % 输出参数：
    %   metrics - 包含各种对比度指标的结构体
    
    % 基本统计对比
    stats_orig = compute_image_stats(img_orig);
    stats_eq = compute_image_stats(img_eq);
    
    metrics.std_improvement = (stats_eq.std - stats_orig.std) / stats_orig.std * 100;
    metrics.range_improvement = (stats_eq.range - stats_orig.range) / stats_orig.range * 100;
    
    % 直方图平坦度指标
    [counts_orig, ~] = custom_histogram(img_orig);
    pdf_orig = counts_orig / sum(counts_orig);
    
    [counts_eq, ~] = custom_histogram(img_eq);
    pdf_eq = counts_eq / sum(counts_eq);
    
    % 与均匀分布的差异（越小越好）
    uniform_pdf = ones(256, 1) / 256;
    metrics.uniformity_diff_orig = sum((pdf_orig - uniform_pdf').^2);
    metrics.uniformity_diff_eq = sum((pdf_eq - uniform_pdf').^2);
    
    % 信息熵变化
    metrics.entropy_change = stats_eq.entropy - stats_orig.entropy;
    
    % 视觉对比度指标（Michelson对比度）
    metrics.michelson_orig = (stats_orig.max - stats_orig.min) / (stats_orig.max + stats_orig.min);
    metrics.michelson_eq = (stats_eq.max - stats_eq.min) / (stats_eq.max + stats_eq.min);
end