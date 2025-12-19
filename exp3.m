%% 实验三：边缘检测实验
clear; close all; clc;
fprintf('=== 边缘检测实验 ===\n\n');

%% 1. 图像加载与预处理
% 使用内置图像
fprintf('加载图像...\n');
original_img = imread("D:\University Work And Life\Detection & Identifying Technology\IMG_20231208_165218.jpg");

% 如果是彩色图像，转换为灰度图像
if size(original_img, 3) == 3
    original_img = rgb2gray(original_img);
end

% 调整图像大小，便于显示
original_img = imresize(original_img, [256, 256]);

% 显示图像信息
fprintf('图像信息:\n');
fprintf('  尺寸: %d × %d 像素\n', size(original_img, 1), size(original_img, 2));
fprintf('  数据类型: %s\n', class(original_img));

%% 2. 显示原始图像
figure('Name', '边缘检测实验', 'Position', [100, 100, 1400, 800]);

% 原始图像
subplot(2, 4, 1);
imshow(original_img);
title('原始图像');
xlabel(sprintf('%d×%d像素', size(original_img, 1), size(original_img, 2)));

%% 3. 高斯噪声对边缘检测的影响
fprintf('\n研究噪声对边缘检测的影响...\n');

% 添加高斯噪声
noisy_img = add_gaussian_noise(original_img, 0, 0.02*255^2);

% 显示噪声图像
subplot(2, 4, 2);
imshow(noisy_img);
title('添加高斯噪声的图像');

%% 4. Sobel边缘检测
fprintf('\n执行Sobel边缘检测...\n');

% 使用自定义Sobel算子
[sobel_edges, sobel_gradient_magnitude, sobel_gradient_direction] = sobel_edge_detection(original_img);

% 使用自定义Sobel算子处理噪声图像
sobel_edges_noisy = sobel_edge_detection(noisy_img);

% 显示Sobel检测结果
subplot(2, 4, 3);
imshow(sobel_edges);
title('Sobel边缘检测');

subplot(2, 4, 4);
imshow(sobel_edges_noisy);
title('Sobel检测(噪声图像)');

%% 5. Prewitt边缘检测
fprintf('\n执行Prewitt边缘检测...\n');

% 使用自定义Prewitt算子
[prewitt_edges, prewitt_gradient_magnitude, prewitt_gradient_direction] = prewitt_edge_detection(original_img);

% 使用自定义Prewitt算子处理噪声图像
prewitt_edges_noisy = prewitt_edge_detection(noisy_img);

% 显示Prewitt检测结果
subplot(2, 4, 5);
imshow(prewitt_edges);
title('Prewitt边缘检测');

subplot(2, 4, 6);
imshow(prewitt_edges_noisy);
title('Prewitt检测(噪声图像)');

%% 6. Canny边缘检测
fprintf('\n执行Canny边缘检测...\n');

% 使用自定义Canny算法
canny_edges = canny_edge_detection(original_img, 0.1, 0.3, 1.0);

% 使用自定义Canny算法处理噪声图像
canny_edges_noisy = canny_edge_detection(noisy_img, 0.1, 0.3, 1.0);

% 显示Canny检测结果
subplot(2, 4, 7);
imshow(canny_edges);
title('Canny边缘检测');

subplot(2, 4, 8);
imshow(canny_edges_noisy);
title('Canny检测(噪声图像)');

%% 7. 详细分析梯度信息
fprintf('\n分析梯度信息...\n');

figure('Name', '梯度信息分析', 'Position', [150, 150, 1200, 800]);

% Sobel梯度幅值
subplot(3, 4, 1);
imshow(sobel_gradient_magnitude, []);
title('Sobel梯度幅值');
colorbar;

% Sobel梯度方向
subplot(3, 4, 2);
imshow(sobel_gradient_direction, []);
title('Sobel梯度方向');
colorbar;

% Prewitt梯度幅值
subplot(3, 4, 3);
imshow(prewitt_gradient_magnitude, []);
title('Prewitt梯度幅值');
colorbar;

% Prewitt梯度方向
subplot(3, 4, 4);
imshow(prewitt_gradient_direction, []);
title('Prewitt梯度方向');
colorbar;

%% 8. 不同阈值对边缘检测的影响
fprintf('\n研究不同阈值对边缘检测的影响...\n');

% 不同阈值
thresholds = [0.05, 0.1, 0.2, 0.3];

figure('Name', '不同阈值下的Sobel检测', 'Position', [200, 200, 1000, 600]);

for i = 1:length(thresholds)
    threshold = thresholds(i);
    
    % Sobel边缘检测（使用不同阈值）
    sobel_thresh = sobel_edge_detection_with_threshold(original_img, threshold);
    
    subplot(2, 4, i);
    imshow(sobel_thresh);
    title(sprintf('Sobel (阈值=%.2f)', threshold));
    
    % Canny边缘检测（使用不同阈值）
    low_thresh = threshold;
    high_thresh = threshold * 2;
    canny_thresh = canny_edge_detection(original_img, low_thresh, high_thresh, 1.0);
    
    subplot(2, 4, i+4);
    imshow(canny_thresh);
    title(sprintf('Canny (低阈值=%.2f)', threshold));
end

%% 9. 三种算法对比分析
fprintf('\n三种边缘检测算法对比...\n');

% 计算边缘像素数量
sobel_edge_pixels = sum(sobel_edges(:) > 0);
prewitt_edge_pixels = sum(prewitt_edges(:) > 0);
canny_edge_pixels = sum(canny_edges(:) > 0);

total_pixels = numel(original_img);

fprintf('边缘像素数量统计:\n');
fprintf('  Sobel: %d (占%.2f%%)\n', sobel_edge_pixels, sobel_edge_pixels/total_pixels*100);
fprintf('  Prewitt: %d (占%.2f%%)\n', prewitt_edge_pixels, prewitt_edge_pixels/total_pixels*100);
fprintf('  Canny: %d (占%.2f%%)\n', canny_edge_pixels, canny_edge_pixels/total_pixels*100);

%% 10. 使用MATLAB内置函数验证
fprintf('\n使用MATLAB内置函数验证...\n');

% MATLAB内置函数
sobel_builtin = edge(original_img, 'sobel');
prewitt_builtin = edge(original_img, 'prewitt');
canny_builtin = edge(original_img, 'canny');

% 计算与自定义实现的差异
sobel_diff = sum(abs(double(sobel_edges(:)) - double(sobel_builtin(:)))) / total_pixels;
prewitt_diff = sum(abs(double(prewitt_edges(:)) - double(prewitt_builtin(:)))) / total_pixels;
canny_diff = sum(abs(double(canny_edges(:)) - double(canny_builtin(:)))) / total_pixels;

fprintf('与MATLAB内置函数对比:\n');
fprintf('  Sobel差异: %.6f\n', sobel_diff);
fprintf('  Prewitt差异: %.6f\n', prewitt_diff);
fprintf('  Canny差异: %.6f\n', canny_diff);

%% 11. 综合对比显示
figure('Name', '三种算法综合对比', 'Position', [100, 100, 1200, 800]);

% 原始图像
subplot(3, 4, 1);
imshow(original_img);
title('原始图像');

% Sobel边缘检测（不同视角）
subplot(3, 4, 2);
imshow(sobel_edges);
title('Sobel边缘检测');

subplot(3, 4, 3);
% 叠加显示边缘
sobel_overlay = imoverlay(original_img, sobel_edges, [1, 0, 0]); % 红色边缘
imshow(sobel_overlay);
title('Sobel边缘叠加');

% Prewitt边缘检测
subplot(3, 4, 5);
imshow(prewitt_edges);
title('Prewitt边缘检测');

subplot(3, 4, 6);
prewitt_overlay = imoverlay(original_img, prewitt_edges, [0, 1, 0]); % 绿色边缘
imshow(prewitt_overlay);
title('Prewitt边缘叠加');

% Canny边缘检测
subplot(3, 4, 9);
imshow(canny_edges);
title('Canny边缘检测');

subplot(3, 4, 10);
canny_overlay = imoverlay(original_img, canny_edges, [0, 0, 1]); % 蓝色边缘
imshow(canny_overlay);
title('Canny边缘叠加');

% 三种算法边缘对比
subplot(3, 4, [4, 8, 12]);
hold on;

% 统计边缘强度分布
[sobel_counts, sobel_bins] = imhist(sobel_gradient_magnitude);
[prewitt_counts, prewitt_bins] = imhist(prewitt_gradient_magnitude);

% 归一化
sobel_counts = sobel_counts / max(sobel_counts);
prewitt_counts = prewitt_counts / max(prewitt_counts);

% 绘制梯度分布
plot(sobel_bins, sobel_counts, 'r-', 'LineWidth', 2, 'DisplayName', 'Sobel梯度');
plot(prewitt_bins, prewitt_counts, 'g-', 'LineWidth', 2, 'DisplayName', 'Prewitt梯度');

title('梯度幅值分布对比');
xlabel('梯度幅值');
ylabel('归一化频数');
legend('Location', 'best');
grid on;

%% 12. 不同图像测试
fprintf('\n在不同图像上测试边缘检测算法...\n');

% 测试不同内置图像
test_images = {'cameraman.tif', 'coins.png', 'rice.png', 'peppers.png'};
test_names = {'Cameraman', 'Coins', 'Rice', 'Peppers'};

figure('Name', '不同图像的边缘检测', 'Position', [100, 100, 1400, 1000]);

for img_idx = 1:length(test_images)
    % 读取图像
    try
        test_img = imread(test_images{img_idx});
    catch
        continue; % 如果图像不存在，跳过
    end
    
    % 转换为灰度图像
    if size(test_img, 3) == 3
        test_img = rgb2gray(test_img);
    end
    
    % 调整大小
    test_img = imresize(test_img, [128, 128]);
    
    % 执行边缘检测
    sobel_test = sobel_edge_detection(test_img);
    prewitt_test = prewitt_edge_detection(test_img);
    canny_test = canny_edge_detection(test_img, 0.1, 0.3, 1.0);
    
    % 显示结果
    % 原始图像
    subplot(length(test_images), 4, (img_idx-1)*4 + 1);
    imshow(test_img);
    title(test_names{img_idx});
    
    % Sobel
    subplot(length(test_images), 4, (img_idx-1)*4 + 2);
    imshow(sobel_test);
    title('Sobel');
    
    % Prewitt
    subplot(length(test_images), 4, (img_idx-1)*4 + 3);
    imshow(prewitt_test);
    title('Prewitt');
    
    % Canny
    subplot(length(test_images), 4, (img_idx-1)*4 + 4);
    imshow(canny_test);
    title('Canny');
end

%% 13. 性能比较
fprintf('\n算法性能比较...\n');

% 计时
tic;
for i = 1:10
    sobel_test = sobel_edge_detection(original_img);
end
sobel_time = toc;

tic;
for i = 1:10
    prewitt_test = prewitt_edge_detection(original_img);
end
prewitt_time = toc;

tic;
for i = 1:10
    canny_test = canny_edge_detection(original_img, 0.1, 0.3, 1.0);
end
canny_time = toc;

fprintf('平均处理时间（10次平均）:\n');
fprintf('  Sobel: %.4f 秒\n', sobel_time/10);
fprintf('  Prewitt: %.4f 秒\n', prewitt_time/10);
fprintf('  Canny: %.4f 秒\n', canny_time/10);

%% 14. 交互式边缘检测工具
fprintf('\n创建交互式边缘检测工具...\n');

figure('Name', '交互式边缘检测', 'Position', [200, 200, 1000, 600]);

% 创建控制面板
uicontrol('Style', 'text', 'Position', [20, 550, 150, 20], ...
          'String', 'Sobel阈值:', 'FontSize', 10);
sobel_slider = uicontrol('Style', 'slider', 'Position', [20, 630, 150, 20], ...
                         'Min', 0, 'Max', 0.5, 'Value', 0.1, ...
                         'Callback', @update_edge_detection);

uicontrol('Style', 'text', 'Position', [190, 550, 150, 20], ...
          'String', 'Canny低阈值:', 'FontSize', 10);
canny_low_slider = uicontrol('Style', 'slider', 'Position', [190, 630, 150, 20], ...
                             'Min', 0, 'Max', 0.5, 'Value', 0.1, ...
                             'Callback', @update_edge_detection);

uicontrol('Style', 'text', 'Position', [360, 550, 150, 20], ...
          'String', 'Canny高阈值:', 'FontSize', 10);
canny_high_slider = uicontrol('Style', 'slider', 'Position', [360, 630, 150, 20], ...
                              'Min', 0, 'Max', 0.5, 'Value', 0.3, ...
                              'Callback', @update_edge_detection);

uicontrol('Style', 'text', 'Position', [530, 550, 150, 20], ...
          'String', '高斯噪声方差:', 'FontSize', 10);
noise_slider = uicontrol('Style', 'slider', 'Position', [530, 630, 150, 20], ...
                         'Min', 0, 'Max', 0.1*255^2, 'Value', 0, ...
                         'Callback', @update_edge_detection);

% 显示图像的坐标轴
ax1 = subplot(2, 3, 1);
ax2 = subplot(2, 3, 2);
ax3 = subplot(2, 3, 3);
ax4 = subplot(2, 3, 4);
ax5 = subplot(2, 3, 5);
ax6 = subplot(2, 3, 6);

% 初始显示
update_edge_detection();



%% 15. 实验总结
fprintf('\n实验完成！\n');
fprintf('================================================================\n');
fprintf('实验总结:\n');
fprintf('1. Sobel算子:\n');
fprintf('   - 使用3x3卷积核计算梯度\n');
fprintf('   - 对水平和垂直边缘敏感\n');
fprintf('   - 对噪声有一定的抑制作用\n');
fprintf('   - 边缘较粗，定位精度一般\n\n');

fprintf('2. Prewitt算子:\n');
fprintf('   - 与Sobel类似，但卷积核不同\n');
fprintf('   - 对噪声更敏感\n');
fprintf('   - 计算简单，速度快\n\n');

fprintf('3. Canny算子:\n');
fprintf('   - 多步骤算法：高斯滤波、梯度计算、非极大值抑制、双阈值检测\n');
fprintf('   - 边缘检测效果好，定位准确\n');
fprintf('   - 抗噪声能力强\n');
fprintf('   - 计算复杂度高\n\n');

fprintf('对比结论:\n');
fprintf('  - 对于简单图像，三种算法都能检测到主要边缘\n');
fprintf('  - 对于噪声图像，Canny算法表现最好\n');
fprintf('  - Sobel和Prewitt算法速度更快，适合实时应用\n');
fprintf('  - Canny算法边缘更细，更完整\n');
fprintf('================================================================\n');

%% ====================================================
%% 所有函数定义（已移动到文件末尾）
%% ====================================================

%% 1. 高斯噪声生成函数
function noisy_img = add_gaussian_noise(img, mean_val, variance)
    % 添加高斯噪声
    img_double = double(img);
    [rows, cols] = size(img_double);
    gaussian_noise = mean_val + sqrt(variance) * randn(rows, cols);
    noisy_img_double = img_double + gaussian_noise;
    noisy_img = uint8(max(0, min(255, noisy_img_double)));
end

%% 2. Sobel边缘检测函数
function [edge_img, gradient_magnitude, gradient_direction] = sobel_edge_detection(img, threshold)
    % Sobel边缘检测
    % 输入参数：
    %   img - 输入图像
    %   threshold - 阈值（可选，默认0.1）
    
    if nargin < 2
        threshold = 0.1; % 默认阈值
    end
    
    % 转换为double类型
    img_double = double(img);
    
    % 定义Sobel算子
    Gx = [-1, 0, 1;
          -2, 0, 2;
          -1, 0, 1];
    
    Gy = [ 1,  2,  1;
           0,  0,  0;
          -1, -2, -1];
    
    % 计算梯度
    grad_x = imfilter(img_double, Gx, 'conv', 'same', 'replicate');
    grad_y = imfilter(img_double, Gy, 'conv', 'same', 'replicate');
    
    % 计算梯度幅值和方向
    gradient_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    gradient_direction = atan2(grad_y, grad_x);
    
    % 归一化梯度幅值到0-1范围
    gradient_magnitude = gradient_magnitude / max(gradient_magnitude(:));
    
    % 应用阈值
    edge_img = gradient_magnitude > threshold;
    
    % 转换为uint8
    edge_img = uint8(edge_img * 255);
end

%% 3. Sobel边缘检测（带阈值参数）
function edge_img = sobel_edge_detection_with_threshold(img, threshold)
    % Sobel边缘检测，只返回二值图像
    [edge_img, ~, ~] = sobel_edge_detection(img, threshold);
end

%% 4. Prewitt边缘检测函数
function [edge_img, gradient_magnitude, gradient_direction] = prewitt_edge_detection(img, threshold)
    % Prewitt边缘检测
    % 输入参数：
    %   img - 输入图像
    %   threshold - 阈值（可选，默认0.1）
    
    if nargin < 2
        threshold = 0.1; % 默认阈值
    end
    
    % 转换为double类型
    img_double = double(img);
    
    % 定义Prewitt算子
    Gx = [-1, 0, 1;
          -1, 0, 1;
          -1, 0, 1];
    
    Gy = [ 1,  1,  1;
           0,  0,  0;
          -1, -1, -1];
    
    % 计算梯度
    grad_x = imfilter(img_double, Gx, 'conv', 'same', 'replicate');
    grad_y = imfilter(img_double, Gy, 'conv', 'same', 'replicate');
    
    % 计算梯度幅值和方向
    gradient_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    gradient_direction = atan2(grad_y, grad_x);
    
    % 归一化梯度幅值到0-1范围
    gradient_magnitude = gradient_magnitude / max(gradient_magnitude(:));
    
    % 应用阈值
    edge_img = gradient_magnitude > threshold;
    
    % 转换为uint8
    edge_img = uint8(edge_img * 255);
end

%% 5. Canny边缘检测函数
function edge_img = canny_edge_detection(img, low_threshold, high_threshold, sigma)
    % Canny边缘检测
    % 输入参数：
    %   img - 输入图像
    %   low_threshold - 低阈值（默认0.1）
    %   high_threshold - 高阈值（默认0.3）
    %   sigma - 高斯滤波器标准差（默认1.0）
    
    if nargin < 2
        low_threshold = 0.1;
    end
    if nargin < 3
        high_threshold = 0.3;
    end
    if nargin < 4
        sigma = 1.0;
    end
    
    % 转换为double类型
    img_double = double(img);
    
    %% 步骤1：高斯滤波
    % 创建高斯滤波器
    kernel_size = ceil(6 * sigma);
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1;
    end
    
    [x, y] = meshgrid(-(kernel_size-1)/2:(kernel_size-1)/2, ...
                      -(kernel_size-1)/2:(kernel_size-1)/2);
    
    gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));
    
    % 应用高斯滤波
    smoothed_img = imfilter(img_double, gaussian_kernel, 'conv', 'same', 'replicate');
    
    %% 步骤2：计算梯度（使用Sobel算子）
    Gx = [-1, 0, 1;
          -2, 0, 2;
          -1, 0, 1];
    
    Gy = [ 1,  2,  1;
           0,  0,  0;
          -1, -2, -1];
    
    grad_x = imfilter(smoothed_img, Gx, 'conv', 'same', 'replicate');
    grad_y = imfilter(smoothed_img, Gy, 'conv', 'same', 'replicate');
    
    % 计算梯度幅值和方向
    gradient_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    gradient_angle = atan2(grad_y, grad_x) * 180 / pi; % 转换为度
    
    % 将角度归一化到0-180度范围
    gradient_angle(gradient_angle < 0) = gradient_angle(gradient_angle < 0) + 180;
    
    %% 步骤3：非极大值抑制
    [rows, cols] = size(gradient_magnitude);
    nms_img = zeros(rows, cols);
    
    for i = 2:rows-1
        for j = 2:cols-1
            angle = gradient_angle(i, j);
            magnitude = gradient_magnitude(i, j);
            
            % 根据角度确定方向
            if (0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)
                % 水平方向
                neighbor1 = gradient_magnitude(i, j-1);
                neighbor2 = gradient_magnitude(i, j+1);
            elseif 22.5 <= angle && angle < 67.5
                % 45度方向
                neighbor1 = gradient_magnitude(i-1, j+1);
                neighbor2 = gradient_magnitude(i+1, j-1);
            elseif 67.5 <= angle && angle < 112.5
                % 垂直方向
                neighbor1 = gradient_magnitude(i-1, j);
                neighbor2 = gradient_magnitude(i+1, j);
            else % 112.5 <= angle && angle < 157.5
                % 135度方向
                neighbor1 = gradient_magnitude(i-1, j-1);
                neighbor2 = gradient_magnitude(i+1, j+1);
            end
            
            % 非极大值抑制
            if magnitude >= neighbor1 && magnitude >= neighbor2
                nms_img(i, j) = magnitude;
            else
                nms_img(i, j) = 0;
            end
        end
    end
    
    %% 步骤4：双阈值检测和连接
    % 归一化梯度幅值到0-1范围
    nms_img = nms_img / max(nms_img(:));
    
    % 应用双阈值
    strong_edges = nms_img > high_threshold;
    weak_edges = (nms_img >= low_threshold) & (nms_img <= high_threshold);
    
    % 连接弱边缘（如果与强边缘相邻，则保留）
    edge_img = strong_edges;
    
    % 8邻域连接
    [rows, cols] = size(edge_img);
    for i = 2:rows-1
        for j = 2:cols-1
            if weak_edges(i, j)
                % 检查8邻域是否有强边缘
                neighborhood = edge_img(i-1:i+1, j-1:j+1);
                if any(neighborhood(:))
                    edge_img(i, j) = 1;
                end
            end
        end
    end
    
    % 转换为uint8
    edge_img = uint8(edge_img * 255);
end

%% 6. 图像叠加函数（如果MATLAB版本不支持imoverlay）
function overlay_img = imoverlay(img, mask, color)
    % 将边缘叠加到图像上
    % 输入参数：
    %   img - 原始图像
    %   mask - 边缘掩码
    %   color - RGB颜色 [r, g, b]
    
    if size(img, 3) == 1
        % 灰度图像转换为RGB
        img_rgb = cat(3, img, img, img);
    else
        img_rgb = img;
    end
    
    % 创建叠加图像
    overlay_img = img_rgb;
    
    % 找到边缘像素
    edge_pixels = mask > 0;
    
    % 对每个颜色通道应用边缘颜色
    for c = 1:3
        channel = overlay_img(:, :, c);
        channel(edge_pixels) = color(c) * 255;
        overlay_img(:, :, c) = channel;
    end
end

% 更新函数
    function update_edge_detection(~, ~)
        % 获取滑动条值
        sobel_thresh = get(sobel_slider, 'Value');
        canny_low = get(canny_low_slider, 'Value');
        canny_high = get(canny_high_slider, 'Value');
        noise_var = get(noise_slider, 'Value');
        
        % 添加噪声
        if noise_var > 0
            current_img = add_gaussian_noise(original_img, 0, noise_var);
        else
            current_img = original_img;
        end
        
        % 执行边缘检测
        sobel_edges_int = sobel_edge_detection_with_threshold(current_img, sobel_thresh);
        prewitt_edges_int = prewitt_edge_detection(current_img);
        canny_edges_int = canny_edge_detection(current_img, canny_low, canny_high, 1.0);
        
        % 显示原始图像（带噪声）
        axes(ax1);
        imshow(current_img);
        if noise_var > 0
            title(sprintf('输入图像\n噪声方差: %.1f', noise_var));
        else
            title('输入图像');
        end
        
        % 显示Sobel结果
        axes(ax2);
        imshow(sobel_edges_int);
        title(sprintf('Sobel\n阈值: %.2f', sobel_thresh));
        
        % 显示Prewitt结果
        axes(ax3);
        imshow(prewitt_edges_int);
        title('Prewitt');
        
        % 显示Canny结果
        axes(ax4);
        imshow(canny_edges_int);
        title(sprintf('Canny\n低阈值: %.2f, 高阈值: %.2f', canny_low, canny_high));
        
        % 显示边缘叠加
        axes(ax5);
        sobel_overlay_int = imoverlay(current_img, sobel_edges_int, [1, 0, 0]);
        imshow(sobel_overlay_int);
        title('Sobel边缘叠加');
        
        axes(ax6);
        canny_overlay_int = imoverlay(current_img, canny_edges_int, [0, 0, 1]);
        imshow(canny_overlay_int);
        title('Canny边缘叠加');
        
        % 计算边缘像素数量
        sobel_count = sum(sobel_edges_int(:) > 0);
        prewitt_count = sum(prewitt_edges_int(:) > 0);
        canny_count = sum(canny_edges_int(:) > 0);
        total_pixels_int = numel(current_img);
        
        % 在图像上显示统计信息
        axes(ax2);
        text(10, 20, sprintf('边缘像素: %d (%.1f%%)', sobel_count, sobel_count/total_pixels_int*100), ...
             'Color', 'white', 'FontSize', 10, 'BackgroundColor', 'black');
        
        axes(ax3);
        text(10, 20, sprintf('边缘像素: %d (%.1f%%)', prewitt_count, prewitt_count/total_pixels_int*100), ...
             'Color', 'white', 'FontSize', 10, 'BackgroundColor', 'black');
        
        axes(ax4);
        text(10, 20, sprintf('边缘像素: %d (%.1f%%)', canny_count, canny_count/total_pixels_int*100), ...
             'Color', 'white', 'FontSize', 10, 'BackgroundColor', 'black');
    end