function Radar_Tracking_GUI_Final()
    % =========================================================================
    % 高频地波雷达目标跟踪 - 交互式分析系统 (终极满分版)
    % 核心特性：
    % 1. 全局代价矩阵贪婪关联 (解决航迹抢夺与断裂)
    % 2. 智能断尾裁剪 (去除卡尔曼滤波发散外推的长尾巴)
    % 3. 生命周期档案库 (保存历史跟丢的真实有效航迹)
    % 4. 空间位移门限过滤 (彻底过滤原地闪烁的海杂波)
    % =========================================================================
    
    %% 1. 数据预处理与参数设定
    try
        S = load('HFRData.mat', 'HFRData');
    catch
        errordlg('未找到 HFRData.mat 数据文件，请确保它在当前目录！', '文件错误');
        return;
    end
    
    validFrames = {};
    for k = 1:length(S.HFRData)
        if ~isempty(S.HFRData{1,k}.xy)
            validFrames{end+1} = S.HFRData{1,k}.xy;
        end
    end

    % 线性卡尔曼滤波基础参数
    Ts = 1.0;
    A = [1 Ts 0 0; 0 1 0 0; 0 0 1 Ts; 0 0 0 1];
    H = [1 0 0 0; 0 0 1 0];
    
    % --- 终极调优超参数 ---
    sig_a = 2;            % 过程噪声：赋予目标一定的机动转弯能力，防止过度平滑
    sig_z = 25;           % 测量噪声：充分包容地波雷达的观测误差，防止航迹碎片化
    
    Q = [Ts^4/4 Ts^3/2 0 0; Ts^3/2 Ts^2 0 0; 0 0 Ts^4/4 Ts^3/2; 0 0 Ts^3/2 Ts^2] * sig_a^2;
    R = diag([sig_z^2, sig_z^2]);
    
    gate_thr = 11.83;     % 马氏距离波门阈值
    max_miss = 4;         % 最大允许漏检次数
    min_trk_len = 10;     % 建档门限：实际有效点数必须 >= 10
    min_disp = 200;       % 位移门限：首尾直线距离必须 > 200米 (专杀海面杂波)

    %% 2. 核心跟踪与状态机引擎
    activeTracks = struct('trackID', {}, 'state', {}, 'P', {}, ...
                        'sPath', {}, 'mPath', {}, 'miss', {});
    archivedTracks = []; 
    currentID = 1;

    for k = 1:length(validFrames)
        Zk = validFrames{k};
        obs = Zk(:, [1 3]); % 提取 x, y
        nObs = size(obs, 1);
        N = length(activeTracks);

        % --- 步骤 A: 时前预测 ---
        pred_x = cell(N, 1); pred_p = cell(N, 1);
        for i = 1:N
            pred_x{i} = A * activeTracks(i).state;
            pred_p{i} = A * activeTracks(i).P * A' + Q;
        end

        % --- 步骤 B: 全局代价矩阵关联 (已修复空矩阵死循环Bug) ---
        assign = nan(1, N); 
        usedObs = false(nObs, 1);
        
        if N > 0 && nObs > 0
            cost_matrix = inf(N, nObs);
            for j = 1:N
                z_p = H * pred_x{j};
                S_inn = H * pred_p{j} * H' + R;
                for i = 1:nObs
                    v = obs(i,:)' - z_p;
                    d = v' / S_inn * v; % 马氏距离
                    if d < gate_thr
                        cost_matrix(j, i) = d;
                    end
                end
            end

            % 贪婪全局最小关联
            while true
                [min_val, linear_idx] = min(cost_matrix(:));
                if isempty(min_val) || isinf(min_val)
                    break; 
                end 
                
                [j_sub, i_sub] = ind2sub(size(cost_matrix), linear_idx);
                assign(j_sub) = i_sub;
                usedObs(i_sub) = true;
                
                % 清除所在行和列，防止重复分配
                cost_matrix(j_sub, :) = inf;
                cost_matrix(:, i_sub) = inf;
            end
        end

        % --- 步骤 C: 量测更新 ---
        for j = 1:N
            if ~isnan(assign(j))
                % 匹配成功，卡尔曼修正
                z = obs(assign(j), :)';
                S_inn = H * pred_p{j} * H' + R;
                K = pred_p{j} * H' / S_inn;
                x_u = pred_x{j} + K * (z - H * pred_x{j});
                p_u = (eye(4) - K * H) * pred_p{j};
                
                activeTracks(j).state = x_u; activeTracks(j).P = p_u; activeTracks(j).miss = 0;
                activeTracks(j).sPath(end+1, :) = [x_u(1), x_u(3)];
                activeTracks(j).mPath(end+1, :) = z';
            else
                % 漏检，惯性外推
                activeTracks(j).state = pred_x{j}; activeTracks(j).P = pred_p{j};
                activeTracks(j).miss = activeTracks(j).miss + 1;
                activeTracks(j).sPath(end+1, :) = (H * pred_x{j})';
                activeTracks(j).mPath(end+1, :) = [NaN, NaN];
            end
        end

        % --- 步骤 D: 航迹生命周期管理 (断尾与杂波过滤) ---
        keep_idx = true(N, 1);
        for j = 1:N
            if activeTracks(j).miss >= max_miss
                keep_idx(j) = false; % 标记死亡，准备移出活跃池
                
                % 【智能断尾】：割掉尾部瞎猜的外推点
                m = activeTracks(j).miss;
                if size(activeTracks(j).sPath, 1) > m
                    activeTracks(j).sPath(end-m+1:end, :) = [];
                    activeTracks(j).mPath(end-m+1:end, :) = [];
                end
                
                % 【双重审核：长度门限 + 位移门限过滤海杂波】
                valid_pts_count = sum(~isnan(activeTracks(j).mPath(:,1)));
                start_pos = activeTracks(j).sPath(1, :);
                end_pos = activeTracks(j).sPath(end, :);
                total_displacement = norm(end_pos - start_pos);
                
                if valid_pts_count >= min_trk_len && total_displacement > min_disp
                    archivedTracks = [archivedTracks; activeTracks(j)];
                end
            end
        end
        activeTracks = activeTracks(keep_idx); 

        % --- 步骤 E: 孤儿点起始新航迹 ---
        for i = 1:nObs
            if ~usedObs(i)
                vx = 0; vy = 0;
                if size(Zk, 2) >= 4, vx = Zk(i,2); vy = Zk(i,4); end
                xi = obs(i,1); yi = obs(i,2);
                
                activeTracks(end+1).trackID = currentID;
                activeTracks(end).state = [xi; vx; yi; vy];
                activeTracks(end).P = diag([sig_z^2, 100, sig_z^2, 100]);
                activeTracks(end).sPath = [xi, yi];
                activeTracks(end).mPath = [xi, yi];
                activeTracks(end).miss = 0;
                currentID = currentID + 1;
            end
        end
    end

    % --- 循环结束后的最终清理 (将存活到最后的航迹存档) ---
    for j = 1:length(activeTracks)
        if activeTracks(j).miss > 0
            m = activeTracks(j).miss;
            if size(activeTracks(j).sPath, 1) > m
                activeTracks(j).sPath(end-m+1:end, :) = [];
                activeTracks(j).mPath(end-m+1:end, :) = [];
            end
        end
        valid_pts_count = sum(~isnan(activeTracks(j).mPath(:,1)));
        start_pos = activeTracks(j).sPath(1, :);
        end_pos = activeTracks(j).sPath(end, :);
        total_displacement = norm(end_pos - start_pos);
        
        if valid_pts_count >= min_trk_len && total_displacement > min_disp
            archivedTracks = [archivedTracks; activeTracks(j)];
        end
    end

    if isempty(archivedTracks)
        msgbox('当前参数下未发现有效长航迹，请检查数据或调低门限！', '分析结果');
        return;
    end
    allIDs = [archivedTracks.trackID];

    %% 3. UI 界面构建 (高颜值交互设计)
    hFig = figure('Name', '高频地波雷达目标跟踪分析系统 (大作业终极版)', ...
                  'NumberTitle', 'off', 'Position', [100, 100, 1000, 650], ...
                  'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'figure');

    leftPanel = uipanel('Parent', hFig, 'Title', '📍 有效航迹控制台', ...
                        'FontSize', 12, 'FontWeight', 'bold',...
                        'BackgroundColor', 'w', 'Position', [0.02 0.05 0.25 0.9]);

    hAxes = axes('Parent', hFig, 'Position', [0.35 0.1 0.6 0.8]);
    xlabel(hAxes, '距离向 X坐标 (m)', 'FontWeight', 'bold', 'FontSize', 11); 
    ylabel(hAxes, '方位向 Y坐标 (m)', 'FontWeight', 'bold', 'FontSize', 11);
    grid(hAxes, 'on'); axis(hAxes, 'equal');

    uicontrol('Parent', leftPanel, 'Style', 'text', 'String', ...
              sprintf('共提取出 %d 条真实目标航迹：', length(allIDs)), ...
              'HorizontalAlignment', 'left', 'BackgroundColor', 'w', ...
              'FontSize', 10, 'Units', 'normalized', 'Position', [0.05 0.88 0.9 0.08]);

    listStr = arrayfun(@(x) sprintf('🚢 目标船只 ID: #%d', x), allIDs, 'UniformOutput', false);

    hListBox = uicontrol('Parent', leftPanel, 'Style', 'listbox', ...
                         'String', listStr, 'FontSize', 11, ...
                         'Units', 'normalized', 'Position', [0.05 0.15 0.9 0.72], ...
                         'Callback', @updatePlot);

    uicontrol('Parent', leftPanel, 'Style', 'pushbutton', 'String', '📸 一键导出报告插图', ...
              'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.94 0.94 0.94], ...
              'Units', 'normalized', 'Position', [0.05 0.02 0.9 0.1], ...
              'Callback', @(~,~) saveas(hFig, sprintf('Track_%d_Analysis.png', allIDs(get(hListBox, 'Value')))));

    % 启动时默认选中第一条航迹进行绘制
    updatePlot(hListBox, []);

    %% 4. 回调函数：执行核心绘图逻辑
    function updatePlot(src, ~)
        idx = get(src, 'Value');
        if isempty(idx), return; end
        tID = allIDs(idx);
        tData = archivedTracks(idx);

        cla(hAxes); hold(hAxes, 'on');

        smoothTrk = tData.sPath;
        rawTrk = tData.mPath;
        validPts = ~any(isnan(rawTrk), 2);

        % 配色设计
        lineColor = [0 0.4470 0.7410];      % 经典科技蓝
        obsColor  = [0.8500 0.3250 0.0980]; % 警示活力橙

        % 1. 绘制雷达散点
        plot(hAxes, rawTrk(validPts,1), rawTrk(validPts,2), 'o', ...
            'MarkerFaceColor', obsColor, 'MarkerEdgeColor', 'w', ...
            'MarkerSize', 7, 'DisplayName', '雷达原始回波点');

        % 2. 绘制滤波平滑航迹
        plot(hAxes, smoothTrk(:,1), smoothTrk(:,2), '-', ...
            'Color', lineColor, 'LineWidth', 2.5, ...
            'DisplayName', 'KF 优化航迹');

        % 3. 标记起点和终点
        plot(hAxes, smoothTrk(1,1), smoothTrk(1,2), 'p', ...
            'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'MarkerSize', 14, ...
            'DisplayName', '航迹起点');
        plot(hAxes, smoothTrk(end,1), smoothTrk(end,2), 's', ...
            'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'MarkerSize', 10, ...
            'DisplayName', '航迹终点');

        % 计算总位移显示在标题中
        disp_len = norm(smoothTrk(end,:) - smoothTrk(1,:));
        
        legend(hAxes, 'Location', 'best', 'FontSize', 11);
        title(hAxes, sprintf('目标跟踪详情分析 - 航迹 ID: %d | 有效点: %d | 总位移: %.1f m', ...
            tID, sum(validPts), disp_len), 'FontSize', 14, 'FontWeight', 'bold');
        
        grid(hAxes, 'on'); 
        set(hAxes, 'GridLineStyle', '--', 'GridAlpha', 0.5, 'LineWidth', 1.2);
        axis(hAxes, 'equal'); 
        hold(hAxes, 'off');
    end
end