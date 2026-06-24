function Radar_Tracking_GUI()
    % 高频地波雷达目标跟踪 - 交互式分析系统
    % 特色：内嵌卡尔曼滤波引擎，提供左中右分栏的交互式航迹分析界面
    
    %% 1. 数据预处理与静默跟踪运算
    % 加载数据
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

    % 参数设定 (CV模型)
    Ts = 1.0;
    A = [1 Ts 0 0; 0 1 0 0; 0 0 1 Ts; 0 0 0 1];
    H = [1 0 0 0; 0 0 1 0];
    sig_a = 1; sig_z = 5;
    Q = [Ts^4/4 Ts^3/2 0 0; Ts^3/2 Ts^2 0 0; 0 0 Ts^4/4 Ts^3/2; 0 0 Ts^3/2 Ts^2] * sig_a^2;
    R = diag([sig_z^2, sig_z^2]);
    gate_thr = 10; max_miss = 5;

    % 启动滤波计算
    targetData = struct('trackID', {}, 'state', {}, 'P', {}, ...
                        'sPath', {}, 'mPath', {}, 'miss', {});
    currentID = 1;

    for k = 1:length(validFrames)
        Zk = validFrames{k};
        obs = Zk(:, [1 3]);
        nObs = size(obs, 1);
        N = length(targetData);

        pred_x = cell(N, 1); pred_p = cell(N, 1);
        for i = 1:N
            pred_x{i} = A * targetData(i).state;
            pred_p{i} = A * targetData(i).P * A' + Q;
        end

        used = false(nObs, 1);
        assign = nan(1, N);
        for j = 1:N
            z_p = H * pred_x{j};
            minD = inf; bIdx = -1;
            for i = 1:nObs
                if used(i), continue; end
                v = obs(i,:)' - z_p;
                S_inn = H * pred_p{j} * H' + R;
                d = v' / S_inn * v;
                if d < gate_thr && d < minD
                    minD = d; bIdx = i;
                end
            end
            if bIdx > 0
                assign(j) = bIdx; used(bIdx) = true;
            end
        end

        for j = 1:N
            if ~isnan(assign(j))
                z = obs(assign(j), :)';
                S_inn = H * pred_p{j} * H' + R;
                K = pred_p{j} * H' / S_inn;
                x_u = pred_x{j} + K * (z - H * pred_x{j});
                p_u = (eye(4) - K * H) * pred_p{j};
                targetData(j).state = x_u; targetData(j).P = p_u; targetData(j).miss = 0;
                targetData(j).sPath(end+1, :) = [x_u(1), x_u(3)];
                targetData(j).mPath(end+1, :) = z';
            else
                targetData(j).state = pred_x{j}; targetData(j).P = pred_p{j};
                targetData(j).miss = targetData(j).miss + 1;
                targetData(j).sPath(end+1, :) = (H * pred_x{j})';
                targetData(j).mPath(end+1, :) = [NaN, NaN];
            end
        end

        for i = 1:nObs
            if ~used(i)
                vx = 0; vy = 0;
                if size(Zk, 2) >= 4, vx = Zk(i,2); vy = Zk(i,4); end
                xi = obs(i,1); yi = obs(i,2);
                targetData(end+1).trackID = currentID;
                targetData(end).state = [xi; vx; yi; vy];
                targetData(end).P = diag([sig_z^2, 100, sig_z^2, 100]);
                targetData(end).sPath = [xi, yi];
                targetData(end).mPath = [xi, yi];
                targetData(end).miss = 0;
                currentID = currentID + 1;
            end
        end
        targetData = targetData([targetData.miss] < max_miss);
    end

    %% 2. 构建美观的自定义 GUI 界面
    % 过滤杂讯：只将跟踪长度大于 5 帧的稳定目标加入列表展示
    validTargetData = targetData(arrayfun(@(x) size(x.sPath,1) > 5, targetData));
    if isempty(validTargetData)
        validTargetData = targetData; 
    end
    allIDs = [validTargetData.trackID];

    % 创建主图窗
    hFig = figure('Name', '高频地波雷达目标跟踪分析系统', ...
                  'NumberTitle', 'off', ...
                  'Position', [100, 100, 950, 600], ...
                  'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'figure');

    % 布局划分：左侧25%面板，右侧绘图区
    leftPanel = uipanel('Parent', hFig, 'Title', '📍 航迹控制台', ...
                        'FontSize', 12, 'FontWeight', 'bold',...
                        'BackgroundColor', 'w', ...
                        'Position', [0.02 0.05 0.25 0.9]);

    hAxes = axes('Parent', hFig, 'Position', [0.35 0.1 0.6 0.8]);
    xlabel(hAxes, '横向距离 (m)'); ylabel(hAxes, '纵向距离 (m)');
    grid(hAxes, 'on'); axis(hAxes, 'equal');

    % 面板文字提示
    uicontrol('Parent', leftPanel, 'Style', 'text', ...
              'String', '请选择要分析的航迹ID：', ...
              'HorizontalAlignment', 'left', 'BackgroundColor', 'w', ...
              'Units', 'normalized', 'Position', [0.05 0.88 0.9 0.08]);

    % 生成列表框内容
    listStr = arrayfun(@(x) sprintf('🎯 跟踪目标 #%d', x), allIDs, 'UniformOutput', false);

    % 列表选择控件
    hListBox = uicontrol('Parent', leftPanel, 'Style', 'listbox', ...
                         'String', listStr, 'FontSize', 11, ...
                         'Units', 'normalized', 'Position', [0.05 0.15 0.9 0.72], ...
                         'Callback', @updatePlot);

    % 导出报告按钮
    uicontrol('Parent', leftPanel, 'Style', 'pushbutton', ...
              'String', '📸 导出当前视图', ...
              'FontSize', 11, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.94 0.94 0.94], ...
              'Units', 'normalized', 'Position', [0.05 0.02 0.9 0.1], ...
              'Callback', @(~,~) saveas(hFig, sprintf('Track_%d_Analysis.png', allIDs(get(hListBox, 'Value')))));

    % 初始化显示第一条航迹
    updatePlot(hListBox, []);

    %% 3. 回调函数：处理点击事件并绘图
    function updatePlot(src, ~)
        idx = get(src, 'Value');
        if isempty(idx), return; end
        tID = allIDs(idx);
        tData = validTargetData(idx);

        cla(hAxes); % 清空右侧画布
        hold(hAxes, 'on');

        smoothTrk = tData.sPath;
        rawTrk = tData.mPath;
        validPts = ~any(isnan(rawTrk), 2);

        % 高级感配色方案
        lineColor = [0 0.4470 0.7410];     % 经典蓝
        obsColor  = [0.8500 0.3250 0.0980]; % 活力橙

        % 1. 绘制原始雷达散点
        plot(hAxes, rawTrk(validPts,1), rawTrk(validPts,2), 'o', ...
            'MarkerFaceColor', obsColor, 'MarkerEdgeColor', 'w', ...
            'MarkerSize', 6, 'DisplayName', '雷达原始量测');

        % 2. 绘制卡尔曼平滑航迹
        plot(hAxes, smoothTrk(:,1), smoothTrk(:,2), '-', ...
            'Color', lineColor, 'LineWidth', 2.5, ...
            'DisplayName', 'KF 平滑航迹');

        % 3. 标记航迹起点和终点
        plot(hAxes, smoothTrk(1,1), smoothTrk(1,2), 'p', ...
            'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'MarkerSize', 12, ...
            'DisplayName', '航迹起始点');
        plot(hAxes, smoothTrk(end,1), smoothTrk(end,2), 's', ...
            'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'MarkerSize', 10, ...
            'DisplayName', '目标最后位置');

        % 图表收尾设置
        legend(hAxes, 'Location', 'best', 'FontSize', 10);
        title(hAxes, sprintf('目标跟踪详情分析 - 航迹 ID: %d', tID), 'FontSize', 14, 'FontWeight', 'bold');
        grid(hAxes, 'on');
        set(hAxes, 'GridLineStyle', '--', 'GridAlpha', 0.5);
        axis(hAxes, 'equal');
        hold(hAxes, 'off');
    end
end