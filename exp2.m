%% 实验二：二维DCT变换实验
clear; close all; clc;
fprintf('=== 二维DCT变换实验 ===\n\n');

%% 1. 图像加载与预处理
% 读取图像
original_img = imread("D:\University Work And Life\Detection & Identifying Technology\IMG_20231208_165218.jpg");  % 使用内置图像

% 如果是彩色图像，转换为灰度图像
if size(original_img, 3) == 3
    original_img = rgb2gray(original_img);
end

% 确保图像尺寸是8的倍数（便于分块）
[rows, cols] = size(original_img);
rows = floor(rows/8) * 8;
cols = floor(cols/8) * 8;
original_img = imresize(original_img(1:rows, 1:cols), [rows, cols]);

% 转换为double类型以便计算
img_double = double(original_img);

% 显示原始图像
figure('Name', 'DCT变换实验', 'Position', [100, 100, 1200, 800]);
subplot(3, 4, 1);
imshow(original_img);
title('原始图像');
xlabel(sprintf('尺寸: %d×%d', rows, cols));

%% 2. 执行DCT变换
fprintf('执行DCT变换...\n');

% 对整幅图像进行DCT变换
img_dct = full_image_dct(img_double);

% 显示DCT系数（取对数增强可视化）
subplot(3, 4, 2);
dct_display = log(abs(img_dct) + 1);  % 加1避免log(0)
imshow(dct_display, []);
title('DCT系数图像（对数显示）');
colorbar;

% 显示DCT系数直方图
subplot(3, 4, 3);
histogram(img_dct(:), 100);
title('DCT系数分布');
xlabel('系数值');
ylabel('频数');
grid on;

%% 3. DCT系数分析（能量分布）

% 分析8x8块的能量分布
fprintf('分析DCT系数能量分布...\n');
N = 8;
energy_dist = analyze_dct_energy(img_dct, N);

% 显示能量分布热图
subplot(3, 4, 4);
imagesc(energy_dist);
title('DCT系数能量分布');
xlabel('水平频率');
ylabel('垂直频率');
colorbar;
colormap(jet);
axis equal tight;

% 显示具体数值
fprintf('DCT系数能量分布（前4×4低频部分）:\n');
for i = 1:min(4, N)
    for j = 1:min(4, N)
        fprintf('%6.2f%% ', energy_dist(i, j));
    end
    fprintf('\n');
end

%% 4. 不同压缩比的重建实验
fprintf('\n进行DCT压缩实验...\n');

compression_ratios = [0.1, 0.25, 0.5, 0.75];  % 保留系数的比例
psnr_values = zeros(length(compression_ratios), 1);
compression_rates = zeros(length(compression_ratios), 1);

for idx = 1:length(compression_ratios)
    ratio = compression_ratios(idx);
    
    % 截断DCT系数
    dct_truncated = truncate_dct(img_dct, ratio);
    
    % 计算压缩率
    non_zero_count = nnz(dct_truncated);
    total_pixels = numel(dct_truncated);
    compression_rates(idx) = (1 - non_zero_count/total_pixels) * 100;
    
    % IDCT重建
    img_reconstructed = full_image_idct(dct_truncated);
    
    % 计算PSNR
    psnr_values(idx) = psnr(uint8(img_reconstructed), original_img);
    
    % 显示结果
    subplot(3, 4, idx+4);
    imshow(uint8(img_reconstructed));
    title(sprintf('保留%.0f%%系数\n压缩率: %.1f%%', ratio*100, compression_rates(idx)));
    
    % 显示截断后的DCT系数
    subplot(3, 4, idx+8);
    dct_display_trunc = log(abs(dct_truncated) + 1);
    imshow(dct_display_trunc, []);
    title(sprintf('截断DCT系数\n(保留%.0f%%)', ratio*100));
    colorbar;
end

%% 5. 量化矩阵应用（模拟JPEG压缩）

fprintf('\n应用JPEG量化矩阵...\n');

% JPEG标准量化矩阵（亮度分量）
jpeg_quant_matrix = [
    16,  11,  10,  16,  24,  40,  51,  61;
    12,  12,  14,  19,  26,  58,  60,  55;
    14,  13,  16,  24,  40,  57,  69,  56;
    14,  17,  22,  29,  51,  87,  80,  62;
    18,  22,  37,  56,  68, 109, 103,  77;
    24,  35,  55,  64,  81, 104, 113,  92;
    49,  64,  78,  87, 103, 121, 120, 101;
    72,  92,  95,  98, 112, 100, 103,  99
];

%% 6. 不同质量因子的JPEG压缩模拟

quality_factors = [10, 25, 50, 75];  % JPEG质量因子
figure('Name', 'JPEG量化效果对比', 'Position', [200, 200, 1200, 600]);

for idx = 1:length(quality_factors)
    qf = quality_factors(idx);
    
    % 应用量化和反量化
    dct_quant = apply_quantization(img_dct, jpeg_quant_matrix, qf);
    dct_dequant = apply_dequantization(dct_quant, jpeg_quant_matrix, qf);
    
    % 重建图像
    img_jpeg = full_image_idct(dct_dequant);
    img_jpeg_uint8 = uint8(max(0, min(255, img_jpeg)));
    
    % 计算PSNR
    psnr_jpeg = psnr(img_jpeg_uint8, original_img);
    
    % 计算非零系数比例（压缩率）
    non_zero_ratio = nnz(dct_quant) / numel(dct_quant) * 100;
    
    % 显示结果
    subplot(2, 4, idx);
    imshow(img_jpeg_uint8);
    title(sprintf('质量因子: %d\nPSNR: %.2f dB\n非零系数: %.1f%%', ...
        qf, psnr_jpeg, non_zero_ratio));
    
    % 显示量化误差
    subplot(2, 4, idx+4);
    error_img = double(original_img) - double(img_jpeg_uint8);
    imshow(error_img, []);
    title(sprintf('重建误差\n均方误差: %.2f', mean(error_img(:).^2)));
    colorbar;
end

%% 7. 单个8×8块详细分析
fprintf('\n分析单个8×8块...\n');

% 选择图像中的一个8×8块
block_row = 100;
block_col = 100;
test_block = img_double(block_row:block_row+7, block_col:block_col+7);

% 计算该块的DCT
block_dct = block_dct2(test_block);

% 显示块分析
figure('Name', '单个8×8块分析', 'Position', [300, 300, 1000, 400]);

% 原始块
subplot(1, 4, 1);
imshow(uint8(test_block));
title('原始8×8块');
axis on;
xticks(1:8);
yticks(1:8);
grid on;

% DCT系数
subplot(1, 4, 2);
imagesc(block_dct);
title('DCT系数');
colorbar;
axis square;
xlabel('水平频率');
ylabel('垂直频率');

% DCT系数数值显示
subplot(1, 4, 3);
text(0.1, 0.9, 'DCT系数矩阵:', 'FontWeight', 'bold');
text_y = 0.8;
for i = 1:8
    coeff_str = sprintf('%7.1f ', block_dct(i, :));
    text(0.1, text_y, coeff_str, 'FontName', 'FixedWidth', 'FontSize', 8);
    text_y = text_y - 0.1;
end
axis off;
title('DCT系数值');

% 能量分布
subplot(1, 4, 4);
block_energy = block_dct.^2;
imagesc(log(block_energy + 1));
title('DCT系数能量(对数)');
colorbar;
axis square;

% 显示能量集中特性
dc_coeff = block_dct(1, 1);
ac_coeffs = block_dct(2:end, 2:end);
energy_dc = dc_coeff^2;
energy_ac = sum(ac_coeffs(:).^2);
energy_total = energy_dc + energy_ac;

fprintf('单个8×8块能量分析:\n');
fprintf('  DC系数能量: %.2f (占%.1f%%)\n', energy_dc, energy_dc/energy_total*100);
fprintf('  AC系数能量: %.2f (占%.1f%%)\n', energy_ac, energy_ac/energy_total*100);

%% 8. 实验结果总结

figure('Name', '实验结果总结', 'Position', [100, 100, 800, 400]);

% 绘制PSNR vs 压缩率曲线
subplot(1, 2, 1);
plot(compression_ratios*100, psnr_values, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('保留系数比例 (%)');
ylabel('PSNR (dB)');
title('PSNR vs 压缩比');
grid on;

% 绘制能量分布曲线
subplot(1, 2, 2);
[rows, cols] = size(img_dct);
N = 8;

% 计算不同频率分量的平均能量
freq_energy = zeros(N, N);
for u = 1:N
    for v = 1:N
        freq_energy(u, v) = mean(mean(img_dct(u:N:rows, v:N:cols).^2));
    end
end

% 按Zigzag顺序排列能量
zigzag_energy = zeros(1, N*N);
zigzag_order = reshape(1:N*N, N, N)';
for k = 1:N*N
    [i, j] = find(zigzag_order == k);
    zigzag_energy(k) = freq_energy(i, j);
end

% 归一化
zigzag_energy = zigzag_energy / max(zigzag_energy);

% 绘制能量衰减曲线
plot(1:N*N, zigzag_energy, 'r-', 'LineWidth', 2);
xlabel('Zigzag扫描序号');
ylabel('归一化能量');
title('DCT系数能量衰减曲线');
grid on;

%% 9. 使用MATLAB内置函数验证

fprintf('\n使用MATLAB内置函数验证...\n');

% 使用MATLAB内置dct2函数
dct_builtin = dct2(img_double);
idct_builtin = idct2(dct_builtin);

% 计算误差
error_custom = sum(abs(img_double(:) - img_reconstructed(:))) / numel(img_double);
error_builtin = sum(abs(img_double(:) - idct_builtin(:))) / numel(img_double);

fprintf('实现验证:\n');
fprintf('  自定义实现平均误差: %.6f\n', error_custom);
fprintf('  内置函数平均误差: %.6f\n', error_builtin);
fprintf('  误差差异: %.6f\n', abs(error_custom - error_builtin));

%% 10. 交互式DCT块探索工具

% 创建交互式界面探索不同块的DCT
fprintf('\n创建交互式DCT探索工具...\n');

figure('Name', '交互式DCT块探索', 'Position', [150, 150, 1000, 500]);

% 显示完整图像，允许点击选择块
subplot(1, 3, 1);
h_img = imshow(original_img);
title('点击图像选择8×8块');
hold on;
h_rect = rectangle('Position', [1, 1, 8, 8], 'EdgeColor', 'r', 'LineWidth', 2);

% 显示选中的块
subplot(1, 3, 2);
h_block = imshow(uint8(zeros(8, 8)));
title('选中的8×8块');

% 显示DCT系数
subplot(1, 3, 3);
h_dct = imagesc(zeros(8, 8));
title('DCT系数');
colorbar;
axis square;

% 设置点击回调函数
set(h_img, 'ButtonDownFcn', @onImageClick);
fprintf('\n实验完成！\n');
    function onImageClick(~, ~)
        % 获取点击位置
        point = get(gca, 'CurrentPoint');
        x = round(point(1, 1));
        y = round(point(1, 2));
        
        % 确保在图像范围内
        x = max(1, min(cols-7, x));
        y = max(1, min(rows-7, y));
        
        % 更新矩形位置
        set(h_rect, 'Position', [x, y, 8, 8]);
        
        % 获取并显示选中的块
        selected_block = img_double(y:y+7, x:x+7);
        set(h_block, 'CData', uint8(selected_block));
        
        % 计算并显示DCT系数
        block_dct_selected = block_dct2(selected_block);
        set(h_dct, 'CData', block_dct_selected);
        title(h_dct.Parent, sprintf('DCT系数\nDC=%.1f', block_dct_selected(1,1)));
        
        % 刷新显示
        drawnow;
    end



%% ====================================================
%% 所有函数定义（已移动到文件末尾）
%% ====================================================

%% 1. 生成DCT变换矩阵
function T = generate_dct_matrix(N)
    % 生成N×N的DCT变换矩阵
    T = zeros(N, N);
    for u = 0:N-1
        for x = 0:N-1
            if u == 0
                T(u+1, x+1) = sqrt(1/N);
            else
                T(u+1, x+1) = sqrt(2/N) * cos((2*x+1)*u*pi/(2*N));
            end
        end
    end
end

%% 2. 块DCT变换函数
function block_dct = block_dct2(block)
    % 对8×8块进行DCT变换
    N = 8;
    T = generate_dct_matrix(N);
    block_dct = T * block * T';
end

%% 3. 块IDCT变换函数
function block_idct = block_idct2(block_dct)
    % 对8×8DCT系数进行IDCT变换
    N = 8;
    T = generate_dct_matrix(N);
    block_idct = T' * block_dct * T;
end

%% 4. 整体图像DCT变换（分块处理）
function img_dct = full_image_dct(img)
    % 对整幅图像进行分块DCT变换
    [rows, cols] = size(img);
    img_dct = zeros(rows, cols);
    
    for i = 1:8:rows
        for j = 1:8:cols
            block = img(i:i+7, j:j+7);
            img_dct(i:i+7, j:j+7) = block_dct2(block);
        end
    end
end

%% 5. 整体图像IDCT变换（分块处理）
function img_idct = full_image_idct(img_dct)
    % 对整幅图像的DCT系数进行分块IDCT变换
    [rows, cols] = size(img_dct);
    img_idct = zeros(rows, cols);
    
    for i = 1:8:rows
        for j = 1:8:cols
            block_dct = img_dct(i:i+7, j:j+7);
            img_idct(i:i+7, j:j+7) = block_idct2(block_dct);
        end
    end
end

%% 6. DCT系数分析函数
function [energy_percentage] = analyze_dct_energy(dct_coeff, N)
    % 分析DCT系数的能量分布
    total_energy = sum(dct_coeff(:).^2);
    energy_percentage = zeros(N, N);
    
    for u = 1:N
        for v = 1:N
            % 计算(u,v)频率系数的能量占比
            energy = sum(sum(dct_coeff(u:N:end, v:N:end).^2));
            energy_percentage(u, v) = energy / total_energy * 100;
        end
    end
end

%% 7. Zigzag掩码生成函数
function mask = zigzag_mask(N, keep_ratio)
    % 生成Zigzag扫描掩码
    mask = zeros(N, N);
    
    % Zigzag扫描顺序
    zigzag_order = [
        1,  2,  6,  7,  15, 16, 28, 29;
        3,  5,  8,  14, 17, 27, 30, 43;
        4,  9,  13, 18, 26, 31, 42, 44;
        10, 12, 19, 25, 32, 41, 45, 54;
        11, 20, 24, 33, 40, 46, 53, 55;
        21, 23, 34, 39, 47, 52, 56, 61;
        22, 35, 38, 48, 51, 57, 60, 62;
        36, 37, 49, 50, 58, 59, 63, 64
    ];
    
    % 计算保留的系数数量
    total_coeffs = N * N;
    keep_count = round(total_coeffs * keep_ratio);
    
    % 生成掩码
    for idx = 1:total_coeffs
        [r, c] = find(zigzag_order == idx);
        if idx <= keep_count
            mask(r, c) = 1;
        else
            mask(r, c) = 0;
        end
    end
end

%% 8. DCT系数截断函数
function dct_truncated = truncate_dct(dct_coeff, keep_ratio)
    % 保留指定比例的低频系数，高频置零
    [rows, cols] = size(dct_coeff);
    dct_truncated = zeros(rows, cols);
    
    % 对每个8×8块进行处理
    for i = 1:8:rows
        for j = 1:8:cols
            block = dct_coeff(i:i+7, j:j+7);
            
            % 创建掩码（只保留Zigzag扫描的前k个系数）
            mask = zigzag_mask(8, keep_ratio);
            block_truncated = block .* mask;
            
            dct_truncated(i:i+7, j:j+7) = block_truncated;
        end
    end
end

%% 9. 量化函数
function dct_quantized = apply_quantization(dct_coeff, q_matrix, quality_factor)
    % 应用量化矩阵
    [rows, cols] = size(dct_coeff);
    dct_quantized = zeros(rows, cols);
    
    % 调整量化矩阵质量
    if quality_factor < 50
        scale = 5000 / quality_factor;
    else
        scale = 200 - quality_factor * 2;
    end
    q_scaled = floor((q_matrix * scale + 50) / 100);
    q_scaled = max(q_scaled, 1);  % 确保不为0
    
    for i = 1:8:rows
        for j = 1:8:cols
            block = dct_coeff(i:i+7, j:j+7);
            block_quantized = round(block ./ q_scaled);
            dct_quantized(i:i+7, j:j+7) = block_quantized;
        end
    end
end

%% 10. 反量化函数
function dct_dequantized = apply_dequantization(dct_quantized, q_matrix, quality_factor)
    % 应用反量化
    [rows, cols] = size(dct_quantized);
    dct_dequantized = zeros(rows, cols);
    
    % 调整量化矩阵质量（与量化时相同）
    if quality_factor < 50
        scale = 5000 / quality_factor;
    else
        scale = 200 - quality_factor * 2;
    end
    q_scaled = floor((q_matrix * scale + 50) / 100);
    q_scaled = max(q_scaled, 1);
    
    for i = 1:8:rows
        for j = 1:8:cols
            block = dct_quantized(i:i+7, j:j+7);
            block_dequantized = block .* q_scaled;
            dct_dequantized(i:i+7, j:j+7) = block_dequantized;
        end
    end
end