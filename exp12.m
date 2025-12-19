%% 实验十二 灰度窗口变换
% 作者：[您的姓名]
% 日期：[实验日期]
% 实验目的：理解灰度窗口变换的工作原理和方法

clear;
close all;
clc;

%% 1. 读取图像并转换为灰度图像

% 读取图像
I = imread("D:\University Work And Life\Detection & Identifying Technology\IMG_20231208_165218.jpg");

% 如果图像是彩色的，转换为灰度图像
if size(I, 3) == 3
    I_gray = rgb2gray(I);
else
    I_gray = I;
end

% 将图像转换为double类型以便处理
I_double = double(I_gray);

% 获取图像尺寸和灰度范围
[rows, cols] = size(I_gray);
gray_min = min(I_gray(:));
gray_max = max(I_gray(:));

fprintf('========== 灰度窗口变换实验 ==========\n');
fprintf('图像信息:\n');
fprintf('  图像大小: %d×%d\n', rows, cols);
fprintf('  灰度范围: [%d, %d]\n', gray_min, gray_max);
fprintf('  平均灰度: %.2f\n', mean(I_gray(:)));

%% 2. 显示原始图像和灰度直方图
figure('Name', '灰度窗口变换实验', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

% 显示原始图像
subplot(3, 4, 1);
imshow(I_gray);
title('原始灰度图像');
xlabel(sprintf('灰度范围: [%d, %d]', gray_min, gray_max));

% 显示原始灰度直方图
subplot(3, 4, 5);
histogram(I_gray, 0:255);
title('原始灰度直方图');
xlabel('灰度值');
ylabel('像素数量');
grid on;
hold on;



%% 4. 设置不同的窗口参数进行实验
% 定义多组窗口参数进行实验
window_params = [
    50, 150;    % 中等宽度窗口
    100, 150;   % 窄窗口
    80, 200;    % 较宽窗口
    30, 100;    % 低灰度窗口
    150, 220;   % 高灰度窗口
    120, 180;   % 中间灰度窗口
];

% 对每组参数进行窗口变换并显示结果
for k = 1:size(window_params, 1)
    L = window_params(k, 1);
    U = window_params(k, 2);
    
    % 应用窗口变换
    I_window = window_transform(I_gray, L, U);
    
    % 显示窗口变换结果
    subplot(3, 4, k+1);
    imshow(I_window);
    title(sprintf('窗口[%d, %d]', L, U));
    
    % 显示窗口变换后的直方图
    subplot(3, 4, k+5);
    histogram(I_window, 0:255);
    title(sprintf('窗口[%d,%d]直方图', L, U));
    xlabel('灰度值');
    ylabel('像素数量');
    grid on;
    
    % 计算统计信息
    black_pixels = sum(I_window(:) == 0);
    white_pixels = sum(I_window(:) == 255);
    window_pixels = sum((I_window(:) > 0) & (I_window(:) < 255));
    total_pixels = rows * cols;
    
    % 显示在直方图上方
    info_str = sprintf('黑:%.1f%%,窗:%.1f%%,白:%.1f%%', ...
        black_pixels/total_pixels*100, ...
        window_pixels/total_pixels*100, ...
        white_pixels/total_pixels*100);
    text(128, max(ylim)*0.9, info_str, ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
    
    % 在原始直方图上标记窗口范围
    subplot(3, 4, 5);
    fill([L, U, U, L], [0, 0, max(ylim)*0.1, max(ylim)*0.1], ...
        [0.8, 0.2, 0.2], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end

% 为原始直方图添加图例
subplot(3, 4, 5);
legend('灰度分布', '窗口范围', 'Location', 'best');

%% 5. 窗口变换函数可视化
% 创建窗口变换函数曲线图
figure('Name', '窗口变换函数曲线', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);

% 生成变换函数曲线
x = 0:255;
L = 80;  % 示例窗口下限
U = 180; % 示例窗口上限

% 根据变换规则计算y值
y = zeros(size(x));
y(x < L) = 0;
y((x >= L) & (x <= U)) = x((x >= L) & (x <= U));
y(x > U) = 255;

% 绘制变换函数
plot(x, y, 'b-', 'LineWidth', 2);
hold on;
grid on;

% 标记窗口范围
fill([L, U, U, L], [0, 0, 255, 255], [0.8, 0.9, 1.0], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
plot([L, L], [0, 255], 'r--', 'LineWidth', 1.5);
plot([U, U], [0, 255], 'r--', 'LineWidth', 1.5);
plot([0, L], [0, 0], 'k-', 'LineWidth', 2);
plot([U, 255], [255, 255], 'k-', 'LineWidth', 2);

% 添加标签和标题
xlabel('输入灰度值');
ylabel('输出灰度值');
title(sprintf('灰度窗口变换函数 (L=%d, U=%d)', L, U));
legend('变换函数', '窗口区域', '窗口边界', 'Location', 'southeast');

% 添加文本说明
text(L/2, 50, 'f(x)=0', 'HorizontalAlignment', 'center', 'FontSize', 12);
text((L+U)/2, 150, 'f(x)=x', 'HorizontalAlignment', 'center', 'FontSize', 12);
text((U+255)/2, 200, 'f(x)=255', 'HorizontalAlignment', 'center', 'FontSize', 12);

%% 6. 交互式窗口变换演示
% 创建一个交互式演示界面
figure('Name', '交互式窗口变换演示', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 400]);

% 定义初始窗口参数
initial_L = 80;
initial_U = 180;

% 创建滑动条控件
subplot(1, 3, 1);
imshow(I_gray);
title('原始图像');

% 创建滑动条控制面板
subplot(1, 3, 2);
% 这里使用uicontrol创建滑动条，但为了简单，我们创建另一个图
% 实际实现中，可以使用GUI工具或App Designer创建交互界面
% 这里我们简单显示一个示例
I_example = window_transform(I_gray, initial_L, initial_U);
imshow(I_example);
title(sprintf('窗口变换结果\nL=%d, U=%d', initial_L, initial_U));

% 显示变换函数
subplot(1, 3, 3);
% 绘制变换函数
x = 0:255;
y = zeros(size(x));
y(x < initial_L) = 0;
y((x >= initial_L) & (x <= initial_U)) = x((x >= initial_L) & (x <= initial_U));
y(x > initial_U) = 255;

plot(x, y, 'b-', 'LineWidth', 2);
hold on;
plot([initial_L, initial_L], [0, 255], 'r--', 'LineWidth', 1.5);
plot([initial_U, initial_U], [0, 255], 'r--', 'LineWidth', 1.5);
xlabel('输入灰度值');
ylabel('输出灰度值');
title('窗口变换函数');
grid on;
xlim([0, 255]);
ylim([0, 255]);

%% 7. 窗口变换与阈值变换的对比
fprintf('\n=== 窗口变换与阈值变换对比 ===\n');

% 创建对比图
figure('Name', '窗口变换与阈值变换对比', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 600]);

% 定义参数
L = 100;
U = 150;
T = 125; % 阈值变换的阈值

% 阈值变换函数
I_threshold = I_gray > T;
I_threshold = uint8(I_threshold * 255);

% 窗口变换
I_window_compare = window_transform(I_gray, L, U);

% 显示原始图像
subplot(2, 4, 1);
imshow(I_gray);
title('原始图像');

% 显示阈值变换结果
subplot(2, 4, 2);
imshow(I_threshold);
title(sprintf('阈值变换\n(T=%d)', T));

% 显示窗口变换结果
subplot(2, 4, 3);
imshow(I_window_compare);
title(sprintf('窗口变换\n(L=%d, U=%d)', L, U));

% 显示两种变换的直方图对比
subplot(2, 4, 5);
histogram(I_gray, 0:255);
title('原始直方图');
xlabel('灰度值');
ylabel('像素数量');
grid on;

subplot(2, 4, 6);
histogram(I_threshold, [0, 128, 255]);
title('阈值变换直方图');
xlabel('灰度值');
ylabel('像素数量');
xlim([0, 255]);
grid on;

subplot(2, 4, 7);
histogram(I_window_compare, 0:255);
title('窗口变换直方图');
xlabel('灰度值');
ylabel('像素数量');
grid on;

% 显示变换函数对比
subplot(2, 4, [4, 8]);
% 阈值变换函数
x = 0:255;
y_threshold = zeros(size(x));
y_threshold(x <= T) = 0;
y_threshold(x > T) = 255;

% 窗口变换函数
y_window = zeros(size(x));
y_window(x < L) = 0;
y_window((x >= L) & (x <= U)) = x((x >= L) & (x <= U));
y_window(x > U) = 255;

plot(x, y_threshold, 'r-', 'LineWidth', 2);
hold on;
plot(x, y_window, 'b-', 'LineWidth', 2);
xlabel('输入灰度值');
ylabel('输出灰度值');
title('变换函数对比');
legend('阈值变换', '窗口变换', 'Location', 'southeast');
grid on;
xlim([0, 255]);
ylim([0, 255]);

%% 8. 图像分割应用示例
fprintf('\n=== 窗口变换在图像分割中的应用 ===\n');

% 创建一个更复杂的图像分割示例
figure('Name', '窗口变换图像分割应用', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);

% 使用不同的窗口进行多层次分割
windows = [
    0, 50;      % 提取暗区域
    50, 100;    % 提取中暗区域
    100, 150;   % 提取中亮区域
    150, 255;   % 提取亮区域
];

% 创建彩色合成图像以显示分割结果
I_color = zeros(rows, cols, 3, 'uint8');

% 为每个窗口分配不同的颜色
colors = [
    255, 0, 0;      % 红色 - 暗区域
    0, 255, 0;      % 绿色 - 中暗区域
    0, 0, 255;      % 蓝色 - 中亮区域
    255, 255, 0;    % 黄色 - 亮区域
];

% 应用多层次窗口分割
for w = 1:size(windows, 1)
    L = windows(w, 1);
    U = windows(w, 2);
    
    % 创建该窗口的二值掩膜
    mask = (I_gray >= L) & (I_gray <= U);
    
    % 将该区域的像素设置为对应颜色
    for c = 1:3
        channel = I_color(:, :, c);
        channel(mask) = colors(w, c);
        I_color(:, :, c) = channel;
    end
    
    % 显示单个窗口分割结果
    subplot(3, 4, w);
    I_single_window = window_transform(I_gray, L, U);
    imshow(I_single_window);
    title(sprintf('窗口[%d,%d]', L, U));
    
    % 显示该窗口的掩膜
    subplot(3, 4, w+4);
    imshow(mask);
    title(sprintf('掩膜[%d,%d]', L, U));
end

% 显示彩色合成结果
subplot(3, 4, [9, 10, 11, 12]);
imshow(I_color);
title('多层次窗口分割合成图');
xlabel('不同颜色表示不同灰度区域');

%% 9. 实验总结与分析
fprintf('\n========== 实验总结 ==========\n');
fprintf('实验目的：理解灰度窗口变换的工作原理和方法\n\n');
fprintf('实验原理总结：\n');
fprintf('  灰度窗口变换的数学表达式：\n');
fprintf('    f(x) = 0;          x < L\n');
fprintf('    f(x) = x;        L ≤ x ≤ U\n');
fprintf('    f(x) = 255;        x > U\n\n');
fprintf('  其中，L表示窗口下限，U表示窗口上限\n\n');
fprintf('窗口变换的特点：\n');
fprintf('  1. 保留了窗口内[L,U]的原始灰度信息\n');
fprintf('  2. 窗口外的灰度被压缩为0或255\n');
fprintf('  3. 可以用于突出显示特定灰度范围的细节\n');
fprintf('  4. 比阈值变换保留了更多信息\n\n');
fprintf('应用场景：\n');
fprintf('  1. 医学图像中特定组织的增强\n');
fprintf('  2. 工业检测中特定灰度范围缺陷的提取\n');
fprintf('  3. 遥感图像中特定地物的分割\n');
fprintf('  4. 多层次图像分割\n');


fprintf('实验完成！\n');
%% 3. 灰度窗口变换函数定义
% 定义窗口变换函数
function output_img = window_transform(input_img, L, U)
    % 灰度窗口变换
    % 输入：
    %   input_img - 输入灰度图像矩阵
    %   L - 窗口下限
    %   U - 窗口上限
    % 输出：
    %   output_img - 变换后的图像
    
    % 初始化输出图像
    output_img = zeros(size(input_img), 'uint8');
    
    % 应用窗口变换规则
    % f(x) = 0;    x < L
    % f(x) = x;    L ≤ x ≤ U
    % f(x) = 255;  x > U
    
    for i = 1:size(input_img, 1)
        for j = 1:size(input_img, 2)
            pixel_value = input_img(i, j);
            
            if pixel_value < L
                output_img(i, j) = 0;
            elseif pixel_value > U
                output_img(i, j) = 255;
            else
                output_img(i, j) = pixel_value;
            end
        end
    end
end