%% 实验十三 视频运动估计
% 作者：[您的姓名]
% 日期：[实验日期]
% 实验目的：理解视频运动估计的工作原理和方法

clear all;
close all;
clc;

fprintf('========== 实验十三 视频运动估计 ==========\n');
fprintf('实验目的：理解视频运动估计的工作原理和方法\n\n');
fprintf('实验原理：\n');
fprintf('  运动估计是视频压缩编码中的核心技术之一\n');
fprintf('  用于消除视频信号的时间冗余，提高编码效率\n');
fprintf('  基本思想是准确获得序列图像帧间的运动位移（运动矢量）\n');

%% 1. 准备测试视频序列
fprintf('\n=== 准备测试视频序列 ===\n');

% 创建合成运动视频序列
fprintf('生成合成运动视频序列...\n');

% 视频参数设置
width = 240;      % 视频宽度（减小以加快处理速度）
height = 180;     % 视频高度（减小以加快处理速度）
num_frames = 20;  % 帧数（减少帧数）
fps = 10;         % 帧率

% 创建视频写入对象
output_video = VideoWriter('test_motion.avi');
output_video.FrameRate = fps;
open(output_video);

% 创建三个不同的运动物体
% 1. 水平移动的矩形
rect1_pos = [50, 50, 30, 30];  % [x, y, width, height]
rect1_speed = [2, 0];          % 水平移动速度

% 2. 垂直移动的圆形
circle_center = [width-80, 80];
circle_radius = 15;
circle_speed = [0, 1.5];       % 垂直移动速度

% 3. 对角线移动的三角形
triangle_pos = [80, height-80]; % 三角形顶点
triangle_speed = [1.5, -1.5];   % 对角线移动

% 生成视频帧
video_frames = zeros(height, width, 3, num_frames, 'uint8');

for frame = 1:num_frames
    % 创建空白帧
    current_frame = zeros(height, width, 3, 'uint8');
    
    % 绘制背景（渐变）
    [X, Y] = meshgrid(1:width, 1:height);
    background = uint8(128 + 64 * sin(X/50 + Y/50 + frame/10));
    current_frame(:, :, 1) = background;
    current_frame(:, :, 2) = background;
    current_frame(:, :, 3) = background;
    
    % 更新并绘制矩形1（水平移动）
    rect1_pos(1) = rect1_pos(1) + rect1_speed(1);
    rect1_pos(2) = rect1_pos(2) + rect1_speed(2);
    
    % 边界检查
    if rect1_pos(1) < 1 || rect1_pos(1) + rect1_pos(3) > width
        rect1_speed(1) = -rect1_speed(1);
    end
    if rect1_pos(2) < 1 || rect1_pos(2) + rect1_pos(4) > height
        rect1_speed(2) = -rect1_speed(2);
    end
    
    % 绘制矩形（红色）
    x1 = max(1, floor(rect1_pos(1)));
    y1 = max(1, floor(rect1_pos(2)));
    x2 = min(width, floor(rect1_pos(1) + rect1_pos(3)));
    y2 = min(height, floor(rect1_pos(2) + rect1_pos(4)));
    
    current_frame(y1:y2, x1:x2, 1) = 255;  % 红色通道
    current_frame(y1:y2, x1:x2, 2) = 0;
    current_frame(y1:y2, x1:x2, 3) = 0;
    
    % 更新并绘制圆形（绿色）
    circle_center = circle_center + circle_speed;
    
    % 边界检查
    if circle_center(1) - circle_radius < 1 || circle_center(1) + circle_radius > width
        circle_speed(1) = -circle_speed(1);
    end
    if circle_center(2) - circle_radius < 1 || circle_center(2) + circle_radius > height
        circle_speed(2) = -circle_speed(2);
    end
    
    % 绘制圆形
    [X, Y] = meshgrid(1:width, 1:height);
    circle_mask = (X - circle_center(1)).^2 + (Y - circle_center(2)).^2 <= circle_radius^2;
    
    current_frame(:, :, 1) = current_frame(:, :, 1) .* uint8(~circle_mask);
    current_frame(:, :, 2) = current_frame(:, :, 2) + uint8(circle_mask) * 255;
    current_frame(:, :, 3) = current_frame(:, :, 3) .* uint8(~circle_mask);
    
    % 更新并绘制三角形（蓝色）
    triangle_pos = triangle_pos + triangle_speed;
    
    % 边界检查
    if triangle_pos(1) < 1 || triangle_pos(1) > width
        triangle_speed(1) = -triangle_speed(1);
    end
    if triangle_pos(2) < 1 || triangle_pos(2) > height
        triangle_speed(2) = -triangle_speed(2);
    end
    
    % 绘制三角形
    triangle_x = [triangle_pos(1), triangle_pos(1)-15, triangle_pos(1)+15];
    triangle_y = [triangle_pos(2), triangle_pos(2)+20, triangle_pos(2)+20];
    
    triangle_mask = poly2mask(triangle_x, triangle_y, height, width);
    
    current_frame(:, :, 1) = current_frame(:, :, 1) .* uint8(~triangle_mask);
    current_frame(:, :, 2) = current_frame(:, :, 2) .* uint8(~triangle_mask);
    current_frame(:, :, 3) = current_frame(:, :, 3) + uint8(triangle_mask) * 255;
    
    % 存储帧
    video_frames(:, :, :, frame) = current_frame;
    
    % 写入视频
    writeVideo(output_video, current_frame);
end

close(output_video);

% 转换为灰度序列
gray_sequence = zeros(height, width, num_frames, 'uint8');
for frame = 1:num_frames
    gray_sequence(:, :, frame) = rgb2gray(video_frames(:, :, :, frame));
end

fprintf('合成视频已生成：test_motion.avi\n');
fprintf('视频大小：%d×%d，%d帧，%.1f fps\n', width, height, num_frames, fps);

%% 2. 块匹配运动估计算法实现
fprintf('\n=== 块匹配运动估计算法实现 ===\n');

% 选择连续两帧进行运动估计
frame1 = gray_sequence(:, :, 5);  % 参考帧
frame2 = gray_sequence(:, :, 6);  % 当前帧

% 显示连续两帧
figure('Name', '连续视频帧', 'NumberTitle', 'off', 'Position', [100, 100, 800, 400]);
subplot(1, 2, 1);
imshow(frame1);
title('参考帧 (Frame 5)');

subplot(1, 2, 2);
imshow(frame2);
title('当前帧 (Frame 6)');

% 块匹配算法参数
block_size = 16;      % 块大小
search_range = 8;     % 搜索范围
method = 'SAD';       % 匹配准则：SAD（绝对误差和）或SSD（平方误差和）

fprintf('块匹配参数：\n');
fprintf('  块大小：%d×%d\n', block_size, block_size);
fprintf('  搜索范围：±%d 像素\n', search_range);
fprintf('  匹配准则：%s\n', method);



% 执行块匹配运动估计
[motion_vectors, compensated_frame] = block_matching(frame1, frame2, block_size, search_range, method);

% 计算残差帧
residual_frame = imabsdiff(frame2, compensated_frame);

%% 3. 显示运动估计结果
figure('Name', '块匹配运动估计结果', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 600]);

% 显示当前帧和补偿帧
subplot(2, 3, 1);
imshow(frame2);
title('当前帧 (原始)');

subplot(2, 3, 2);
imshow(compensated_frame);
title('运动补偿帧');

subplot(2, 3, 3);
imshow(residual_frame);
title('残差帧');
colorbar;

% 显示运动矢量场
subplot(2, 3, 4);
imshow(frame1);
hold on;

% 绘制运动矢量
[num_blocks_h, num_blocks_w, ~] = size(motion_vectors);

% 创建网格
[X, Y] = meshgrid(1:num_blocks_w, 1:num_blocks_h);

% 计算块中心位置
center_x = (X(:) - 1) * block_size + block_size/2;
center_y = (Y(:) - 1) * block_size + block_size/2;

% 获取运动矢量
dx = motion_vectors(:,:,2);
dy = motion_vectors(:,:,1);

% 确保向量维度匹配
center_x = center_x(:);
center_y = center_y(:);
dx = dx(:);
dy = dy(:);

% 绘制运动矢量
quiver(center_x, center_y, dx, dy, 0, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
title('运动矢量场');
hold off;

% 显示运动矢量幅度和角度
subplot(2, 3, 5);
% 计算运动矢量幅度
mv_magnitude = sqrt(dx.^2 + dy.^2);
imagesc(reshape(mv_magnitude, num_blocks_h, num_blocks_w));
colorbar;
title('运动矢量幅度');
xlabel('块列索引');
ylabel('块行索引');
axis image;

subplot(2, 3, 6);
% 计算运动矢量方向（角度）
mv_angle = atan2(dy, dx) * 180 / pi;
imagesc(reshape(mv_angle, num_blocks_h, num_blocks_w));
colorbar;
title('运动矢量方向（角度）');
xlabel('块列索引');
ylabel('块行索引');
axis image;

%% 4. 不同块大小和搜索范围的比较
fprintf('\n=== 不同参数对运动估计的影响 ===\n');

% 测试不同的块大小
block_sizes = [8, 16, 32];
search_ranges = [4, 8, 16];

% 创建比较图
figure('Name', '不同参数比较', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

plot_idx = 1;
for bs = block_sizes
    for sr = search_ranges
        % 执行运动估计
        [mv_test, comp_test] = block_matching(frame1, frame2, bs, sr, 'SAD');
        
        % 计算残差
        residual_test = imabsdiff(frame2, comp_test);
        
        % 计算残差能量（用于评估运动估计效果）
        residual_energy = sum(residual_test(:).^2);
        
        % 显示结果
        subplot(length(block_sizes), length(search_ranges), plot_idx);
        imshow(comp_test);
        title(sprintf('块大小:%d, 搜索:%d\n残差能量:%.2e', bs, sr, residual_energy));
        
        plot_idx = plot_idx + 1;
    end
end

%% 5. 不同匹配准则的比较
fprintf('\n=== 不同匹配准则的比较 ===\n');

% 使用不同匹配准则
methods = {'SAD', 'SSD'};

figure('Name', '不同匹配准则比较', 'NumberTitle', 'off', 'Position', [100, 100, 800, 400]);

for m = 1:length(methods)
    % 执行运动估计
    [mv_method, comp_method] = block_matching(frame1, frame2, block_size, search_range, methods{m});
    
    % 计算残差
    residual_method = imabsdiff(frame2, comp_method);
    
    % 计算残差能量
    residual_energy = sum(residual_method(:).^2);
    
    % 显示结果
    subplot(1, 2, m);
    imshow(residual_method);
    title(sprintf('%s匹配准则\n残差能量:%.2e', methods{m}, residual_energy));
    colorbar;
end

%% 6. 多帧运动估计和运动轨迹跟踪
fprintf('\n=== 多帧运动估计和轨迹跟踪 ===\n');

% 选择多帧进行运动估计
start_frame = 5;
end_frame = 10;  % 减少帧数
num_track_frames = end_frame - start_frame + 1;

% 跟踪特定块的运动轨迹
track_block_row = 3;  % 要跟踪的块行索引
track_block_col = 3;  % 要跟踪的块列索引

% 存储轨迹
trajectory = zeros(num_track_frames, 2);  % [x, y]

% 多帧运动估计
figure('Name', '多帧运动轨迹跟踪', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 600]);

for f = 1:num_track_frames
    frame_idx = start_frame + f - 1;
    
    if frame_idx < num_frames
        % 获取连续两帧
        ref_frame_multi = gray_sequence(:, :, frame_idx);
        cur_frame_multi = gray_sequence(:, :, frame_idx + 1);
        
        % 执行运动估计
        [mv_multi, ~] = block_matching(ref_frame_multi, cur_frame_multi, block_size, search_range, 'SAD');
        
        % 记录特定块的运动矢量
        if f == 1
            % 初始位置（块中心）
            start_x = (track_block_col - 1) * block_size + block_size/2;
            start_y = (track_block_row - 1) * block_size + block_size/2;
            trajectory(f, :) = [start_x, start_y];
        end
        
        % 更新位置
        if f < num_track_frames
            dx_multi = mv_multi(track_block_row, track_block_col, 2);
            dy_multi = mv_multi(track_block_row, track_block_col, 1);
            
            trajectory(f+1, 1) = trajectory(f, 1) + dx_multi;
            trajectory(f+1, 2) = trajectory(f, 2) + dy_multi;
        end
        
        % 显示当前帧和运动矢量
        subplot(2, 3, f);
        imshow(cur_frame_multi);
        hold on;
        
        % 绘制运动矢量场（简化版）
        [num_blocks_h_multi, num_blocks_w_multi, ~] = size(mv_multi);
        [X_multi, Y_multi] = meshgrid(1:num_blocks_w_multi, 1:num_blocks_h_multi);
        center_x_multi = (X_multi(:) - 1) * block_size + block_size/2;
        center_y_multi = (Y_multi(:) - 1) * block_size + block_size/2;
        dx_all = mv_multi(:,:,2);
        dy_all = mv_multi(:,:,1);
        
        % 确保向量维度匹配
        center_x_multi = center_x_multi(:);
        center_y_multi = center_y_multi(:);
        dx_all = dx_all(:);
        dy_all = dy_all(:);
        
        % 只绘制部分运动矢量以避免拥挤
        step = 2;
        quiver(center_x_multi(1:step:end), center_y_multi(1:step:end), ...
               dx_all(1:step:end), dy_all(1:step:end), 0, 'r', 'LineWidth', 1);
        
        % 标记跟踪的块
        plot(trajectory(f, 1), trajectory(f, 2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
        
        title(sprintf('帧 %d', frame_idx+1));
        hold off;
    end
end

% 绘制运动轨迹
figure('Name', '运动轨迹', 'NumberTitle', 'off');
imshow(gray_sequence(:, :, start_frame));
hold on;
plot(trajectory(:, 1), trajectory(:, 2), 'r-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
title(sprintf('块(%d,%d)的运动轨迹', track_block_row, track_block_col));
xlabel('X坐标');
ylabel('Y坐标');
legend('运动轨迹', 'Location', 'best');
grid on;

%% 7. 运动估计在视频压缩中的应用演示
fprintf('\n=== 运动估计在视频压缩中的应用 ===\n');

% 演示基于运动估计的视频压缩思想
figure('Name', '视频压缩原理演示', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 500]);

% 选择一组连续帧
demo_frames = gray_sequence(:, :, 5:7);

% 参考帧
subplot(2, 3, 1);
imshow(demo_frames(:, :, 1));
title('参考帧 (I帧)');

% 当前帧
subplot(2, 3, 2);
imshow(demo_frames(:, :, 2));
title('当前帧 (P帧)');

% 运动补偿帧
[mv_demo, comp_demo] = block_matching(demo_frames(:, :, 1), demo_frames(:, :, 2), 16, 8, 'SAD');
subplot(2, 3, 3);
imshow(comp_demo);
title('运动补偿预测帧');

% 残差帧
residual_demo = imabsdiff(demo_frames(:, :, 2), comp_demo);
subplot(2, 3, 4);
imshow(residual_demo);
title('残差帧');
colorbar;

% 原始帧数据量
original_size = numel(demo_frames(:, :, 2)) * 8;  % 假设8位/像素

% 估计压缩后的数据量
% 假设：运动矢量 + 残差 = 压缩数据
mv_bits = numel(mv_demo) * 10;  % 假设每个运动矢量用10位表示
residual_nonzero = sum(residual_demo(:) > 5);  % 只编码明显的残差
residual_bits = residual_nonzero * 8;  % 假设每个残差用8位表示

compressed_size = mv_bits + residual_bits;
compression_ratio = original_size / compressed_size;

% 显示压缩信息
subplot(2, 3, [5, 6]);
bar([1, 2], [original_size/1000, compressed_size/1000]);
set(gca, 'XTickLabel', {'原始帧', '压缩数据'});
ylabel('数据量 (千比特)');
title(sprintf('压缩比: %.2f:1', compression_ratio));
grid on;

text(1, original_size/1000*0.9, sprintf('%.1f kb', original_size/1000), ...
    'HorizontalAlignment', 'center', 'FontSize', 10);
text(2, compressed_size/1000*0.9, sprintf('%.1f kb', compressed_size/1000), ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

fprintf('视频压缩演示：\n');
fprintf('  原始帧数据量：%.2f kb\n', original_size/1000);
fprintf('  压缩后数据量：%.2f kb\n', compressed_size/1000);
fprintf('  压缩比：%.2f:1\n', compression_ratio);

%% 8. 实验总结与算法分析
fprintf('\n========== 实验总结 ==========\n');
fprintf('实验目的：理解视频运动估计的工作原理和方法\n\n');
fprintf('实验内容总结：\n');
fprintf('  1. 实现了块匹配运动估计算法\n');
fprintf('  2. 比较了不同块大小和搜索范围的影响\n');
fprintf('  3. 比较了SAD和SSD匹配准则\n');
fprintf('  4. 实现了多帧运动轨迹跟踪\n');
fprintf('  5. 演示了运动估计在视频压缩中的应用\n\n');
fprintf('块匹配算法特点：\n');
fprintf('  优点：\n');
fprintf('    - 算法简单，易于实现\n');
fprintf('    - 计算复杂度相对较低\n');
fprintf('    - 适合硬件实现\n');
fprintf('  缺点：\n');
fprintf('    - 假设块内所有像素运动一致\n');
fprintf('    - 对旋转、缩放等复杂运动处理不佳\n');
fprintf('    - 块效应明显\n\n');
fprintf('运动估计算法分类：\n');
fprintf('  1. 块匹配法（本实验实现）\n');
fprintf('  2. 递归估计法\n');
fprintf('  3. 贝叶斯估计法\n');
fprintf('  4. 光流法\n\n');
fprintf('应用领域：\n');
fprintf('  1. 视频压缩编码（MPEG, H.264, HEVC）\n');
fprintf('  2. 视频稳定和去抖动\n');
fprintf('  3. 动作识别和行为分析\n');
fprintf('  4. 目标跟踪和监控\n');
fprintf('  5. 计算机视觉和机器人导航\n');
fprintf('实验完成！\n');

%% 实现块匹配运动估计算法
function [motion_vectors, compensated_frame] = block_matching(ref_frame, cur_frame, block_size, search_range, method)
    % 块匹配运动估计算法
    % 输入：
    %   ref_frame - 参考帧
    %   cur_frame - 当前帧
    %   block_size - 块大小
    %   search_range - 搜索范围
    %   method - 匹配准则 ('SAD' 或 'SSD')
    % 输出：
    %   motion_vectors - 运动矢量场
    %   compensated_frame - 运动补偿后的帧
    
    [height, width] = size(ref_frame);
    
    % 计算块的数量
    num_blocks_h = floor(height / block_size);
    num_blocks_w = floor(width / block_size);
    
    % 初始化运动矢量场
    motion_vectors = zeros(num_blocks_h, num_blocks_w, 2); % [dy, dx]
    
    % 初始化运动补偿帧
    compensated_frame = zeros(height, width, 'uint8');
    
    % 对每个块进行匹配
    for i = 1:num_blocks_h
        for j = 1:num_blocks_w
            % 参考块位置
            ref_y = (i-1) * block_size + 1;
            ref_x = (j-1) * block_size + 1;
            
            % 提取参考块
            ref_block = ref_frame(ref_y:ref_y+block_size-1, ref_x:ref_x+block_size-1);
            
            % 搜索范围边界
            min_y = max(1, ref_y - search_range);
            max_y = min(height - block_size + 1, ref_y + search_range);
            min_x = max(1, ref_x - search_range);
            max_x = min(width - block_size + 1, ref_x + search_range);
            
            % 初始化最小误差和最佳匹配位置
            min_error = Inf;
            best_dy = 0;
            best_dx = 0;
            
            % 在搜索范围内搜索最佳匹配
            for dy = min_y:max_y
                for dx = min_x:max_x
                    % 提取候选块
                    cand_block = cur_frame(dy:dy+block_size-1, dx:dx+block_size-1);
                    
                    % 计算匹配误差
                    if strcmp(method, 'SAD')
                        % 绝对误差和
                        error = sum(abs(double(ref_block(:)) - double(cand_block(:))));
                    else
                        % 平方误差和
                        error = sum((double(ref_block(:)) - double(cand_block(:))).^2);
                    end
                    
                    % 更新最佳匹配
                    if error < min_error
                        min_error = error;
                        best_dy = dy;
                        best_dx = dx;
                    end
                end
            end
            
            % 计算运动矢量
            motion_vectors(i, j, 1) = best_dy - ref_y;  % dy
            motion_vectors(i, j, 2) = best_dx - ref_x;  % dx
            
            % 运动补偿：用参考块填充补偿帧
            compensated_frame(ref_y:ref_y+block_size-1, ref_x:ref_x+block_size-1) = ...
                cur_frame(best_dy:best_dy+block_size-1, best_dx:best_dx+block_size-1);
        end
    end
    
    % 转换为uint8
    compensated_frame = uint8(compensated_frame);
end