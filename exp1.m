%% 实验一：噪声图像的产生实验（修改版）
clear; close all; clc;
fprintf('=== 噪声图像产生实验 ===\n\n');

%% 1. 图像加载模块
% 使用内置图像确保代码可运行
fprintf('使用MATLAB内置图像...\n');
original_img = imread("D:\University Work And Life\Detection & Identifying Technology\IMG_20231208_165218.jpg");

% 如果是彩色图像，转换为灰度图像
if size(original_img, 3) == 3
    original_img = rgb2gray(original_img);
end

% 显示图像基本信息
fprintf('图像信息:\n');
fprintf('  尺寸: %d x %d 像素\n', size(original_img, 1), size(original_img, 2));
fprintf('  数据类型: %s\n', class(original_img));
fprintf('  像素值范围: [%d, %d]\n', min(original_img(:)), max(original_img(:)));

%% 2. 噪声生成函数定义（所有函数定义在文件末尾）

%% 3. 生成各种噪声图像
fprintf('\n生成噪声图像...\n');

% 高斯噪声参数
mean_val = 0;
variance_low = 0.01 * 255^2;   % 低方差
variance_high = 0.1 * 255^2;    % 高方差

% 椒盐噪声参数
density_low = 0.02;            % 低密度
density_high = 0.1;            % 高密度
salt_prob = 0.5;               % 盐噪声比例

% 生成单一噪声图像
gaussian_low = add_gaussian_noise(original_img, mean_val, variance_low);
gaussian_high = add_gaussian_noise(original_img, mean_val, variance_high);
sp_noise_low = add_salt_pepper_noise(original_img, density_low, salt_prob);
sp_noise_high = add_salt_pepper_noise(original_img, density_high, salt_prob);
pepper_noise = add_salt_pepper_noise(original_img, 0.05, 0);   % 纯胡椒噪声

% 生成叠加噪声图像（高斯噪声 + 椒盐噪声）
fprintf('生成叠加噪声图像...\n');
% 先添加高斯噪声
combined_gaussian = add_gaussian_noise(original_img, mean_val, variance_low);
% 再添加椒盐噪声
combined_noise = add_salt_pepper_noise(combined_gaussian, 0.05, 0.5);

%% 4. 显示结果
figure('Name', '噪声图像对比', 'Position', [100, 100, 1200, 800]);

% 原始图像
subplot(3, 3, 1);
imshow(original_img);
title('原始图像', 'FontSize', 10);

% 高斯噪声图像
subplot(3, 3, 2);
imshow(gaussian_low);
title('高斯噪声(低方差)', 'FontSize', 10);

subplot(3, 3, 3);
imshow(gaussian_high);
title('高斯噪声(高方差)', 'FontSize', 10);

% 椒盐噪声图像
subplot(3, 3, 4);
imshow(sp_noise_low);
title('椒盐噪声(低密度)', 'FontSize', 10);

subplot(3, 3, 5);
imshow(sp_noise_high);
title('椒盐噪声(高密度)', 'FontSize', 10);

subplot(3, 3, 6);
imshow(pepper_noise);
title('纯胡椒噪声', 'FontSize', 10);

% 叠加噪声图像（替换原来的纯盐噪声）
subplot(3, 3, 7);
imshow(combined_noise);
title('叠加噪声(高斯+椒盐)', 'FontSize', 10);

%% 5. 分析叠加噪声的成分
% 提取高斯噪声成分
gaussian_component = double(gaussian_low) - double(original_img);

% 提取椒盐噪声成分
salt_pepper_component = double(sp_noise_low) - double(original_img);

% 显示噪声成分
subplot(3, 3, 8);
imshow(gaussian_component, []);
title('高斯噪声成分', 'FontSize', 10);
colorbar;

subplot(3, 3, 9);
imshow(salt_pepper_component, []);
title('椒盐噪声成分', 'FontSize', 10);
colorbar;

% 添加总标题
sgtitle('噪声图像产生实验 - 结果对比', 'FontSize', 14, 'FontWeight', 'bold');

%% 6. 直方图分析
figure('Name', '噪声图像直方图分析', 'Position', [150, 150, 1200, 800]);

% 原始图像直方图
subplot(3, 3, 1);
imhist(original_img);
title('原始图像直方图', 'FontSize', 10);

% 高斯噪声图像直方图
subplot(3, 3, 2);
imhist(gaussian_low);
title('高斯噪声直方图', 'FontSize', 10);

subplot(3, 3, 3);
imhist(gaussian_high);
title('高斯噪声(高方差)直方图', 'FontSize', 10);

% 椒盐噪声图像直方图
subplot(3, 3, 4);
imhist(sp_noise_low);
title('椒盐噪声(低密度)直方图', 'FontSize', 10);

subplot(3, 3, 5);
imhist(sp_noise_high);
title('椒盐噪声(高密度)直方图', 'FontSize', 10);

subplot(3, 3, 6);
imhist(pepper_noise);
title('纯胡椒噪声直方图', 'FontSize', 10);

% 叠加噪声直方图
subplot(3, 3, 7);
imhist(combined_noise);
title('叠加噪声直方图', 'FontSize', 10);

% 直方图对比
subplot(3, 3, [8, 9]);
hold on;
h1 = histogram(double(original_img(:)), 'BinWidth', 5, 'FaceAlpha', 0.5, 'DisplayName', '原始图像');
h2 = histogram(double(combined_noise(:)), 'BinWidth', 5, 'FaceAlpha', 0.5, 'DisplayName', '叠加噪声');
title('原始图像与叠加噪声直方图对比', 'FontSize', 10);
xlabel('像素强度');
ylabel('频数');
legend;
grid on;

sgtitle('噪声图像直方图分析', 'FontSize', 14, 'FontWeight', 'bold');

%% 7. 噪声图像统计特性分析
fprintf('\n=== 噪声图像统计特性分析 ===\n\n');



% 计算各种噪声图像的质量指标
noise_types = {'高斯噪声(低方差)', '高斯噪声(高方差)', ...
               '椒盐噪声(低密度)', '椒盐噪声(高密度)', '纯胡椒噪声', '叠加噪声'};
noise_images = {gaussian_low, gaussian_high, ...
                sp_noise_low, sp_noise_high, pepper_noise, combined_noise};

fprintf('%-25s %12s %12s\n', '噪声类型', 'SNR(dB)', 'PSNR(dB)');
fprintf('%s\n', repmat('-', 1, 50));

for i = 1:length(noise_images)
    snr_val = calculate_snr(original_img, noise_images{i});
    psnr_val = calculate_psnr(original_img, noise_images{i});
    fprintf('%-25s %10.2f %12.2f\n', noise_types{i}, snr_val, psnr_val);
end

%% 8. 噪声可视化与3D视图
figure('Name', '噪声三维可视化', 'Position', [200, 200, 1000, 600]);

% 原始图像表面
subplot(2, 3, 1);
surf(double(original_img(1:4:end, 1:4:end)), 'EdgeColor', 'none');
title('原始图像表面', 'FontSize', 10);
view(3);
colormap(jet);
colorbar;

% 高斯噪声表面
subplot(2, 3, 2);
gaussian_noise_only = double(gaussian_low) - double(original_img);
surf(gaussian_noise_only(1:4:end, 1:4:end), 'EdgeColor', 'none');
title('高斯噪声表面', 'FontSize', 10);
view(3);
colormap(jet);
colorbar;

% 椒盐噪声表面
subplot(2, 3, 3);
sp_noise_only = double(sp_noise_low) - double(original_img);
surf(sp_noise_only(1:4:end, 1:4:end), 'EdgeColor', 'none');
title('椒盐噪声表面', 'FontSize', 10);
view(3);
colormap(jet);
colorbar;

% 叠加噪声表面
subplot(2, 3, 4);
combined_noise_only = double(combined_noise) - double(original_img);
surf(combined_noise_only(1:4:end, 1:4:end), 'EdgeColor', 'none');
title('叠加噪声表面', 'FontSize', 10);
view(3);
colormap(jet);
colorbar;

% 叠加噪声强度分布
subplot(2, 3, [5, 6]);
hold on;

% 绘制不同噪声的强度分布
x = 1:size(original_img, 2);
y_original = double(original_img(100, :));
y_gaussian = double(gaussian_low(100, :));
y_sp = double(sp_noise_low(100, :));
y_combined = double(combined_noise(100, :));

plot(x, y_original, 'k-', 'LineWidth', 2, 'DisplayName', '原始图像');
plot(x, y_gaussian, 'b-', 'LineWidth', 1, 'DisplayName', '高斯噪声');
plot(x, y_sp, 'r-', 'LineWidth', 1, 'DisplayName', '椒盐噪声');
plot(x, y_combined, 'g-', 'LineWidth', 2, 'DisplayName', '叠加噪声');

title('第100行像素强度分布', 'FontSize', 10);
xlabel('列位置');
ylabel('像素强度');
legend('Location', 'best');
grid on;


%% 10. 实验总结
fprintf('\n实验完成！\n');
fprintf('================================================================\n');
fprintf('实验总结:\n');
fprintf('1. 成功生成了高斯噪声、椒盐噪声及其叠加噪声\n');
fprintf('2. 高斯噪声使图像整体变模糊，直方图呈现高斯分布\n');
fprintf('3. 椒盐噪声在图像上产生随机黑白点，直方图两端有尖峰\n');
fprintf('4. 叠加噪声结合了两种噪声的特性，图像质量进一步下降\n');
fprintf('5. 可以通过调整参数控制噪声的强度\n');
fprintf('================================================================\n');

%% ====================================================
%% 所有函数定义（已移动到文件末尾）
%% ====================================================


% 更新函数
function update_combined_noise(~, ~)
    % 获取滑动条值
    gauss_var = get(gauss_slider, 'Value');
    sp_density = get(sp_slider, 'Value');
    salt_ratio = get(salt_slider, 'Value');
    
    % 生成高斯噪声
    gauss_img = add_gaussian_noise(original_img, 0, gauss_var);
    
    % 生成椒盐噪声
    sp_img = add_salt_pepper_noise(original_img, sp_density, salt_ratio);
    
    % 生成叠加噪声
    combined = add_salt_pepper_noise(gauss_img, sp_density, salt_ratio);
    
    % 更新显示
    axes(ax1);
    imshow(gauss_img);
    title(sprintf('高斯噪声\n方差=%.1f', gauss_var));
    
    axes(ax2);
    imshow(sp_img);
    title(sprintf('椒盐噪声\n密度=%.3f', sp_density));
    
    axes(ax3);
    imshow(combined);
    title(sprintf('叠加噪声\n方差=%.1f, 密度=%.3f', gauss_var, sp_density));
    
    % 显示直方图
    axes(ax4);
    imhist(gauss_img);
    title('高斯噪声直方图');
    
    axes(ax5);
    imhist(sp_img);
    title('椒盐噪声直方图');
    
    axes(ax6);
    imhist(combined);
    title('叠加噪声直方图');
    
    % 计算并显示质量指标
    snr_combined = calculate_snr(original_img, combined);
    psnr_combined = calculate_psnr(original_img, combined);
    
    % 在图像上添加文本
    axes(ax3);
    text(10, 20, sprintf('SNR: %.2f dB', snr_combined), ...
         'Color', 'white', 'FontSize', 10, 'BackgroundColor', 'black');
    text(10, 40, sprintf('PSNR: %.2f dB', psnr_combined), ...
         'Color', 'white', 'FontSize', 10, 'BackgroundColor', 'black');
end

% 计算信噪比(SNR)函数
function snr_val = calculate_snr(original, noisy)
    original = double(original);
    noisy = double(noisy);
    signal_power = mean(original(:).^2);
    noise_power = mean((noisy(:) - original(:)).^2);
    snr_val = 10 * log10(signal_power / noise_power);
end

% 计算PSNR函数
function psnr_val = calculate_psnr(original, noisy)
    original = double(original);
    noisy = double(noisy);
    mse = mean((original(:) - noisy(:)).^2);
    psnr_val = 10 * log10(255^2 / mse);
end

%% 1. 高斯噪声生成函数
function noisy_img = add_gaussian_noise(img, mean_val, variance)
    % 添加高斯噪声
    % 输入参数：
    %   img - 原始图像
    %   mean_val - 噪声均值
    %   variance - 噪声方差
    
    % 将图像转换为double类型以便计算
    img_double = double(img);
    
    % 生成与图像大小相同的高斯噪声矩阵
    [rows, cols] = size(img_double);
    gaussian_noise = mean_val + sqrt(variance) * randn(rows, cols);
    
    % 将噪声叠加到原图像
    noisy_img_double = img_double + gaussian_noise;
    
    % 将像素值限制在0-255范围内，并转换回uint8
    noisy_img = uint8(max(0, min(255, noisy_img_double)));
end

%% 2. 椒盐噪声生成函数
function noisy_img = add_salt_pepper_noise(img, noise_density, salt_prob)
    % 添加椒盐噪声
    % 输入参数：
    %   img - 原始图像
    %   noise_density - 噪声密度（0-1）
    %   salt_prob - 盐噪声概率（0-1）
    
    % 获取图像尺寸
    [rows, cols] = size(img);
    noisy_img = img;
    
    % 生成随机矩阵
    rand_matrix = rand(rows, cols);
    
    % 计算胡椒噪声和盐噪声的阈值
    pepper_threshold = noise_density * (1 - salt_prob);
    salt_threshold = noise_density * salt_prob;
    
    % 添加胡椒噪声（黑色像素，值为0）
    pepper_mask = rand_matrix < pepper_threshold;
    noisy_img(pepper_mask) = 0;
    
    % 添加盐噪声（白色像素，值为255）
    salt_mask = rand_matrix > (1 - salt_threshold);
    noisy_img(salt_mask) = 255;
end