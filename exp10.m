% ===========================================
% 实验十：直方图增强
% 功能：读取一张图像并进行直方图均衡化
% ===========================================
clear all;
close all;
clc;

fprintf('============ 实验十：直方图均衡化 ============\n');

% ============ 第一步：读取图像 ============
% 提示用户选择图像文件
[fname, pname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', ...
                           '图像文件 (*.jpg, *.jpeg, *.png, *.bmp, *.tif, *.tiff)'}, ...
                           '选择要处理的图像');
if isequal(fname, 0) || isequal(pname, 0)
    fprintf('未选择图像，使用默认图像...\n');
    % 使用MATLAB自带的示例图像
    try
        original_img = imread('cameraman.tif');
        fprintf('使用默认图像: cameraman.tif\n');
    catch
        % 如果默认图像不存在，创建一个示例图像
        fprintf('创建示例图像...\n');
        original_img = uint8(randn(256, 256) * 20 + 128);
        original_img = min(max(original_img, 0), 255);
    end
else
    % 读取用户选择的图像
    filepath = fullfile(pname, fname);
    fprintf('正在读取图像: %s\n', fname);
    original_img = imread(filepath);
end

% ============ 第二步：图像预处理 ============
% 如果图像是彩色的，转换为灰度图像
if size(original_img, 3) == 3
    fprintf('检测到彩色图像，转换为灰度图像...\n');
    gray_img = rgb2gray(original_img);
else
    gray_img = original_img;
end

% 确保图像是uint8类型
if ~isa(gray_img, 'uint8')
    fprintf('将图像转换为uint8类型...\n');
    if max(gray_img(:)) <= 1
        gray_img = uint8(gray_img * 255);
    else
        gray_img = uint8(gray_img);
    end
end

% 显示图像信息
[m, n] = size(gray_img);
fprintf('图像信息:\n');
fprintf('  尺寸: %d × %d 像素\n', m, n);
fprintf('  灰度范围: %d ~ %d\n', min(gray_img(:)), max(gray_img(:)));
fprintf('  灰度级数: %d\n', max(gray_img(:)) - min(gray_img(:)) + 1);

% ============ 第三步：计算原始图像直方图 ============
fprintf('\n计算原始图像直方图...\n');

% 计算直方图
histogram = zeros(256, 1);
for i = 1:m
    for j = 1:n
        gray_value = gray_img(i, j) + 1;  % MATLAB索引从1开始
        histogram(gray_value) = histogram(gray_value) + 1;
    end
end

% 计算统计信息
total_pixels = m * n;
min_gray = min(gray_img(:));
max_gray = max(gray_img(:));
mean_gray = mean(double(gray_img(:)));
std_gray = std(double(gray_img(:)));

fprintf('直方图统计信息:\n');
fprintf('  总像素数: %d\n', total_pixels);
fprintf('  平均灰度值: %.2f\n', mean_gray);
fprintf('  灰度标准差: %.2f\n', std_gray);

% ============ 第四步：直方图均衡化 ============
fprintf('\n执行直方图均衡化...\n');

% 方法1：使用自定义函数进行均衡化（便于理解原理）
fprintf('使用自定义函数进行均衡化...\n');

% 计算概率密度函数(PDF)
pdf = histogram / total_pixels;

% 计算累积分布函数(CDF)
cdf = zeros(256, 1);
cdf(1) = pdf(1);
for k = 2:256
    cdf(k) = cdf(k-1) + pdf(k);
end

% 创建灰度映射表
gray_map = round(255 * cdf);

% 应用映射得到均衡化后的图像
equalized_img = zeros(m, n, 'uint8');
for i = 1:m
    for j = 1:n
        gray_value = gray_img(i, j) + 1;
        equalized_img(i, j) = gray_map(gray_value);
    end
end

% ============ 第五步：计算均衡化后图像的直方图 ============
fprintf('计算均衡化后图像的直方图...\n');

histogram_eq = zeros(256, 1);
for i = 1:m
    for j = 1:n
        gray_value = equalized_img(i, j) + 1;
        histogram_eq(gray_value) = histogram_eq(gray_value) + 1;
    end
end

% 计算均衡化后的统计信息
mean_gray_eq = mean(double(equalized_img(:)));
std_gray_eq = std(double(equalized_img(:)));

fprintf('均衡化后统计信息:\n');
fprintf('  平均灰度值: %.2f\n', mean_gray_eq);
fprintf('  灰度标准差: %.2f\n', std_gray_eq);
fprintf('  对比度提升: %.2f%%\n', (std_gray_eq - std_gray) / std_gray * 100);

% ============ 第六步：显示结果 ============
fprintf('\n显示处理结果...\n');

% 创建主显示窗口
figure('Name', '直方图均衡化结果', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1000, 600], 'Color', [0.9, 0.9, 0.9]);

% 1. 显示原始图像
subplot(2, 3, 1);
imshow(gray_img);
title('原始图像', 'FontSize', 10, 'FontWeight', 'bold');
colorbar;
% 添加文本标注
text(10, 20, sprintf('尺寸: %dx%d', m, n), ...
     'Color', 'white', 'FontSize', 8, 'FontWeight', 'bold', ...
     'BackgroundColor', 'black');
text(10, 40, sprintf('灰度: %d-%d', min_gray, max_gray), ...
     'Color', 'white', 'FontSize', 8, 'FontWeight', 'bold', ...
     'BackgroundColor', 'black');

% 2. 显示原始图像直方图
subplot(2, 3, 2);
bar(0:255, histogram, 'FaceColor', [0.2, 0.4, 0.6], 'EdgeColor', 'none');
xlim([0, 255]);
xlabel('灰度级', 'FontSize', 9);
ylabel('像素数', 'FontSize', 9);
title('原始图像直方图', 'FontSize', 10, 'FontWeight', 'bold');
grid on;

% 在直方图上标注统计信息
text(180, max(histogram)*0.9, sprintf('均值: %.1f', mean_gray), ...
     'FontSize', 8, 'BackgroundColor', 'white');
text(180, max(histogram)*0.8, sprintf('标准差: %.1f', std_gray), ...
     'FontSize', 8, 'BackgroundColor', 'white');

% 3. 显示原始图像的累积直方图
subplot(2, 3, 3);
cdf_plot = cumsum(histogram) / total_pixels;
plot(0:255, cdf_plot, 'b-', 'LineWidth', 2);
xlim([0, 255]);
ylim([0, 1]);
xlabel('灰度级', 'FontSize', 9);
ylabel('累积概率', 'FontSize', 9);
title('原始图像累积直方图', 'FontSize', 10, 'FontWeight', 'bold');
grid on;

% 4. 显示均衡化后的图像
subplot(2, 3, 4);
imshow(equalized_img);
title('均衡化后图像', 'FontSize', 10, 'FontWeight', 'bold');
colorbar;
% 添加文本标注
text(10, 20, sprintf('均值: %.1f', mean_gray_eq), ...
     'Color', 'white', 'FontSize', 8, 'FontWeight', 'bold', ...
     'BackgroundColor', 'black');
text(10, 40, sprintf('标准差: %.1f', std_gray_eq), ...
     'Color', 'white', 'FontSize', 8, 'FontWeight', 'bold', ...
     'BackgroundColor', 'black');

% 5. 显示均衡化后图像的直方图
subplot(2, 3, 5);
bar(0:255, histogram_eq, 'FaceColor', [0.6, 0.2, 0.4], 'EdgeColor', 'none');
xlim([0, 255]);
xlabel('灰度级', 'FontSize', 9);
ylabel('像素数', 'FontSize', 9);
title('均衡化后直方图', 'FontSize', 10, 'FontWeight', 'bold');
grid on;

% 在直方图上标注统计信息
text(180, max(histogram_eq)*0.9, sprintf('均值: %.1f', mean_gray_eq), ...
     'FontSize', 8, 'BackgroundColor', 'white');
text(180, max(histogram_eq)*0.8, sprintf('标准差: %.1f', std_gray_eq), ...
     'FontSize', 8, 'BackgroundColor', 'white');

% 6. 显示映射曲线和均衡化后累积直方图
subplot(2, 3, 6);
% 绘制映射曲线
plot(0:255, gray_map, 'r-', 'LineWidth', 2);
hold on;
% 绘制均衡化后图像的累积直方图
cdf_eq_plot = cumsum(histogram_eq) / total_pixels;
plot(0:255, cdf_eq_plot, 'b-', 'LineWidth', 1.5);
% 绘制参考线（对角线）
plot([0, 255], [0, 255], 'k--', 'LineWidth', 1);
hold off;

xlim([0, 255]);
ylim([0, 255]);
xlabel('原始灰度级', 'FontSize', 9);
ylabel('映射后灰度级/累积概率×255', 'FontSize', 9);
title('映射曲线和均衡化后CDF', 'FontSize', 10, 'FontWeight', 'bold');
legend('映射曲线', '均衡化CDF×255', '恒等映射', 'Location', 'southeast');
grid on;

% ============ 第七步：添加对比显示 ============
% 创建对比显示窗口
figure('Name', '对比显示', 'NumberTitle', 'off', ...
       'Position', [150, 150, 800, 400], 'Color', [0.95, 0.95, 0.95]);

% 并排显示原始图像和均衡化后图像
subplot(1, 2, 1);
imshow(gray_img);
title('原始图像', 'FontSize', 11, 'FontWeight', 'bold');

% 在图像上叠加直方图信息
annotation('textbox', [0.15, 0.05, 0.3, 0.1], ...
           'String', sprintf('对比度: %.1f', std_gray), ...
           'FontSize', 9, 'FontWeight', 'bold', ...
           'BackgroundColor', 'white', 'EdgeColor', 'black');

subplot(1, 2, 2);
imshow(equalized_img);
title('直方图均衡化后', 'FontSize', 11, 'FontWeight', 'bold');

% 在图像上叠加直方图信息
annotation('textbox', [0.65, 0.05, 0.3, 0.1], ...
           'String', sprintf('对比度: %.1f', std_gray_eq), ...
           'FontSize', 9, 'FontWeight', 'bold', ...
           'BackgroundColor', 'white', 'EdgeColor', 'black');

% ============ 第八步：保存结果 ============
% 询问是否保存结果
choice = questdlg('是否保存处理结果？', '保存结果', ...
                  '保存均衡化图像', '保存所有结果', '不保存', '保存均衡化图像');

if strcmp(choice, '保存均衡化图像') || strcmp(choice, '保存所有结果')
    % 生成文件名
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    
    % 保存均衡化后的图像
    eq_filename = sprintf('equalized_%s.png', timestamp);
    imwrite(equalized_img, eq_filename);
    fprintf('已保存均衡化图像: %s\n', eq_filename);
    
    if strcmp(choice, '保存所有结果')
        % 保存原始图像
        orig_filename = sprintf('original_%s.png', timestamp);
        imwrite(gray_img, orig_filename);
        fprintf('已保存原始图像: %s\n', orig_filename);
        
        % 保存直方图数据到文本文件
        hist_filename = sprintf('histogram_data_%s.txt', timestamp);
        fid = fopen(hist_filename, 'w');
        fprintf(fid, '灰度级,原始直方图,均衡化直方图\n');
        for k = 1:256
            fprintf(fid, '%d,%d,%d\n', k-1, histogram(k), histogram_eq(k));
        end
        fclose(fid);
        fprintf('已保存直方图数据: %s\n', hist_filename);
    end
end

% ============ 第九步：显示统计摘要 ============
fprintf('\n============ 处理结果摘要 ============\n');
fprintf('原始图像:\n');
fprintf('  灰度范围: %d - %d\n', min_gray, max_gray);
fprintf('  平均灰度: %.2f\n', mean_gray);
fprintf('  标准差: %.2f\n', std_gray);
fprintf('\n均衡化后图像:\n');
fprintf('  灰度范围: %d - %d\n', min(equalized_img(:)), max(equalized_img(:)));
fprintf('  平均灰度: %.2f\n', mean_gray_eq);
fprintf('  标准差: %.2f\n', std_gray_eq);
fprintf('\n对比度提升: %.2f%%\n', (std_gray_eq - std_gray) / std_gray * 100);

% ============ 第十步：完成提示 ============
fprintf('\n============ 直方图均衡化完成 ============\n');
fprintf('程序功能总结:\n');
fprintf('1. 读取图像（支持多种格式）\n');
fprintf('2. 自动转换为灰度图像\n');
fprintf('3. 计算原始图像直方图\n');
fprintf('4. 执行直方图均衡化\n');
fprintf('5. 显示处理前后的图像和直方图\n');
fprintf('6. 显示映射曲线\n');
fprintf('7. 提供统计信息\n');
fprintf('8. 可选保存处理结果\n');

% 显示完成对话框
msgbox('直方图均衡化处理完成！', '完成', 'help');