%% 实验十一 灰度阈值处理
% 作者：[您的姓名]
% 日期：[实验日期]
% 实验目的：理解灰度阈值处理的原理和方法

clear;
close all;
clc;

%% 1. 读取图像并转换为灰度图像
I = imread("D:\University Work And Life\Detection & Identifying Technology\IMG_20231208_165218.jpg");

% 如果图像是彩色的，转换为灰度图像
if size(I, 3) == 3
    I_gray = rgb2gray(I);
else
    I_gray = I;
end

% 显示原始图像
figure('Name', '灰度阈值处理实验', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

subplot(2, 3, 1);
imshow(I_gray);
title('原始灰度图像');
xlabel(sprintf('图像大小: %d×%d', size(I_gray, 1), size(I_gray, 2)));

% 显示灰度直方图
subplot(2, 3, 4);
imhist(I_gray);
title('灰度直方图');
xlabel('灰度值');
ylabel('像素数量');
grid on;

%% 2. 手动选择阈值进行二值化处理
% 设定不同的阈值进行实验
thresholds = [50, 100, 150]; % 三个不同的阈值

for i = 1:length(thresholds)
    T = thresholds(i);
    
    % 进行阈值处理：大于阈值的置为255，小于等于的置为0
    I_binary = I_gray > T;
    
    % 转换为uint8类型（0和255）
    I_binary = uint8(I_binary * 255);
    
    % 显示二值化结果
    subplot(2, 3, i+1);
    imshow(I_binary);
    title(sprintf('阈值 T = %d', T));
    
    % 计算统计信息
    white_pixels = sum(I_binary(:) == 255);
    black_pixels = sum(I_binary(:) == 0);
    total_pixels = numel(I_binary);
    white_ratio = white_pixels / total_pixels * 100;
    
    fprintf('阈值 T = %d:\n', T);
    fprintf('  白色像素(255): %d (%.2f%%)\n', white_pixels, white_ratio);
    fprintf('  黑色像素(0): %d (%.2f%%)\n', black_pixels, 100-white_ratio);
    fprintf('  总像素数: %d\n\n', total_pixels);
end

%% 3. 使用OTSU方法自动确定最佳阈值
% OTSU方法（大津法）是一种自适应的阈值确定方法
% 它会找到一个阈值使得前景和背景的类间方差最大

T_otsu = graythresh(I_gray); % 返回归一化的阈值（0-1之间）
T_otsu_actual = T_otsu * 255; % 转换为实际的灰度值

fprintf('=== OTSU方法 ===\n');
fprintf('归一化阈值: %.4f\n', T_otsu);
fprintf('实际阈值: %.2f\n', T_otsu_actual);

% 使用OTSU阈值进行二值化
I_otsu = imbinarize(I_gray, T_otsu);
I_otsu_display = uint8(I_otsu * 255);

% 显示OTSU结果
subplot(2, 3, 6);
imshow(I_otsu_display);
title(sprintf('OTSU阈值 T = %.1f', T_otsu_actual));

% 计算OTSU结果的统计信息
white_pixels_otsu = sum(I_otsu(:) == 1);
black_pixels_otsu = sum(I_otsu(:) == 0);
white_ratio_otsu = white_pixels_otsu / numel(I_otsu) * 100;

fprintf('OTSU二值化结果:\n');
fprintf('  白色像素(前景): %d (%.2f%%)\n', white_pixels_otsu, white_ratio_otsu);
fprintf('  黑色像素(背景): %d (%.2f%%)\n', black_pixels_otsu, 100-white_ratio_otsu);

%% 4. 展示不同阈值的处理效果对比
figure('Name', '不同阈值对比', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);

% 测试更多阈值
test_thresholds = [30, 80, 130, 180, 230];
num_thresholds = length(test_thresholds);

for i = 1:num_thresholds
    T = test_thresholds(i);
    I_binary = uint8((I_gray > T) * 255);
    
    subplot(2, 3, i);
    imshow(I_binary);
    title(sprintf('T = %d', T));
    xlabel(sprintf('%.1f%%白像素', sum(I_binary(:)==255)/numel(I_binary)*100));
end

% 添加直方图和阈值线
subplot(2, 3, 6);
histogram(double(I_gray(:)), 0:255);
hold on;
for i = 1:num_thresholds
    line([test_thresholds(i), test_thresholds(i)], ylim, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
end
title('灰度直方图与阈值线');
xlabel('灰度值');
ylabel('像素数量');
legend('灰度分布', '阈值位置', 'Location', 'best');
grid on;

%% 5. 阈值处理函数实现（自定义函数）
fprintf('\n=== 自定义阈值函数演示 ===\n');



% 使用自定义函数
T_custom = 100;
I_custom = custom_threshold(I_gray, T_custom);

figure('Name', '自定义函数结果', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(I_custom);
title(sprintf('自定义阈值函数结果 (T=%d)', T_custom));

% 与MATLAB内置函数对比
subplot(1, 2, 2);
I_matlab = uint8((I_gray > T_custom) * 255);
imshow(I_matlab);
title(sprintf('MATLAB内置函数结果 (T=%d)', T_custom));

% 验证两者结果是否一致
if isequal(I_custom, I_matlab)
    fprintf('自定义函数与MATLAB内置函数结果一致！\n');
else
    fprintf('警告：自定义函数与MATLAB内置函数结果有差异！\n');
end

%% 6. 图像分割应用示例
fprintf('\n=== 图像分割应用 ===\n');

% 创建示例：提取图像中的亮区域
figure('Name', '图像分割应用', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 400]);

% 原始图像
subplot(1, 3, 1);
imshow(I_gray);
title('原始图像');

% 选择分割阈值（这里根据直方图选择）
T_segment = 150;
I_segment = I_gray > T_segment;

% 创建一个彩色图像来突出显示分割区域
I_color = repmat(I_gray, [1, 1, 3]); % 转换为彩色图像
red_channel = I_color(:, :, 1);
green_channel = I_color(:, :, 2);
blue_channel = I_color(:, :, 3);

% 将分割区域标记为红色
red_channel(I_segment) = 255;
green_channel(I_segment) = 0;
blue_channel(I_segment) = 0;

I_color(:, :, 1) = red_channel;
I_color(:, :, 2) = green_channel;
I_color(:, :, 3) = blue_channel;

% 显示分割结果
subplot(1, 3, 2);
imshow(I_color);
title(sprintf('分割区域标记 (T=%d)', T_segment));

% 显示二值掩膜
subplot(1, 3, 3);
imshow(I_segment);
title('二值分割掩膜');
colormap(gray);

%% 7. 实验总结
fprintf('\n========== 实验总结 ==========\n');
fprintf('实验目的：理解灰度阈值处理的原理和方法\n');
fprintf('实验内容：\n');
fprintf('  1. 实现了基本的灰度阈值处理算法\n');
fprintf('  2. 比较了不同阈值对分割效果的影响\n');
fprintf('  3. 实现了OTSU自适应阈值方法\n');
fprintf('  4. 展示了图像分割的实际应用\n');
fprintf('\n结论：\n');
fprintf('  1. 阈值选择对二值化效果影响显著\n');
fprintf('  2. 阈值过小会导致过多背景被误判为前景\n');
fprintf('  3. 阈值过大会导致前景信息丢失\n');
fprintf('  4. OTSU方法能自动找到相对较优的阈值\n');
fprintf('  5. 灰度阈值处理是图像分割的基础方法\n');

% 定义自定义阈值函数
function binary_img = custom_threshold(gray_img, T)
    % 灰度阈值处理函数
    % 输入：
    %   gray_img - 灰度图像矩阵
    %   T - 阈值
    % 输出：
    %   binary_img - 二值图像（0和255）
    
    % 获取图像大小
    [rows, cols] = size(gray_img);
    
    % 初始化输出图像
    binary_img = zeros(rows, cols, 'uint8');
    
    % 应用阈值
    for i = 1:rows
        for j = 1:cols
            if gray_img(i, j) > T
                binary_img(i, j) = 255;  % 大于阈值设为255（白色）
            else
                binary_img(i, j) = 0;    % 小于等于阈值设为0（黑色）
            end
        end
    end
end