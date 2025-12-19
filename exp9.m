% 扩展实验：部分取反和自定义取反曲线
clear all;
close all;
clc;

% 1. 部分取反（只取反特定灰度范围的像素）
img = imread("D:\University Work And Life\Detection & Identifying Technology\IMG_20231208_165218.jpg");
figure('Name', '部分取反实验', 'NumberTitle', 'off');

% 原始图像
subplot(2, 3, 1);
imshow(img);
title('原始图像');

% 全取反
full_inverted = 255 - img;
subplot(2, 3, 2);
imshow(full_inverted);
title('全取反');

% 部分取反：只取反低灰度区域（0-127）
partial_inverted = img;
low_pixels = img < 128;
partial_inverted(low_pixels) = 255 - img(low_pixels);
subplot(2, 3, 3);
imshow(partial_inverted);
title('部分取反（低灰度区域）');

% 2. 自定义取反曲线
subplot(2, 3, 4);
r = 0:255;
% 线性取反
s_linear = 255 - r;
plot(r, s_linear, 'b-', 'LineWidth', 2);
hold on;
% 非线性取反（S形曲线）
s_nonlinear = 255 * (1 - sin(pi * r / (2 * 255)));
plot(r, s_nonlinear, 'r-', 'LineWidth', 2);
% 分段取反
s_piecewise = zeros(1, 256);
s_piecewise(1:128) = 255 - 2 * r(1:128);
s_piecewise(129:256) = 2 * (255 - r(129:256));
plot(r, s_piecewise, 'g-', 'LineWidth', 2);
legend('线性取反', '非线性取反', '分段取反');
xlabel('原始像素值 r');
ylabel('取反后像素值 s');
title('不同取反曲线');
grid on;

% 3. 应用非线性取反
nonlinear_inverted = uint8(255 * (1 - sin(pi * double(img) / (2 * 255))));
subplot(2, 3, 5);
imshow(nonlinear_inverted);
title('非线性取反结果');

% 4. 彩色图像不同通道取反
color_img = imread('peppers.png');
% 只取反红色通道
red_inverted = color_img;
red_inverted(:, :, 1) = 255 - color_img(:, :, 1);
subplot(2, 3, 6);
imshow(red_inverted);
title('只取反红色通道');