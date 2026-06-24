% =========================================================================
% 高频地波雷达多目标跟踪算法实现 (主脚本)
% 核心算法：常速(CV)模型 + 卡尔曼滤波(KF) + 最近邻数据关联(NNDA)
% =========================================================================
clear; clc; close all;

%% 1. 雷达数据加载与预处理
load('HFRData.mat', 'HFRData');  

% 剔除空数据帧，保留有效的雷达探测点迹
validMeasurements = {};
for frameIdx = 1:length(HFRData)
    currentXY = HFRData{1, frameIdx}.xy;
    if ~isempty(currentXY)
        validMeasurements{end+1} = currentXY;
    end
end
totalFrames = length(validMeasurements);

%% 2. 跟踪系统状态空间参数初始化
T_sample = 1.0; % 雷达扫描周期/采样间隔

% 状态转移矩阵 (4x4, CV模型状态向量: [x位置, x速度, y位置, y速度])
TransMat = [1 T_sample 0 0;
            0 1        0 0;
            0 0        1 T_sample;
            0 0        0 1];
        
% 观测矩阵 (2x4, 雷达仅能观测到 [x位置, y位置])
ObsMat = [1 0 0 0;
          0 0 1 0];

% 系统噪声参数设置
std_accel = 1; % 过程噪声标准差 (假设的未知加速度)
std_meas = 5;  % 测量噪声标准差 (雷达定位误差)

% 过程噪声协方差矩阵 Q
Q_noise = [T_sample^4/4, T_sample^3/2, 0,            0;
           T_sample^3/2, T_sample^2,   0,            0;
           0,            0,            T_sample^4/4, T_sample^3/2;
           0,            0,            T_sample^3/2, T_sample^2] * (std_accel^2);

% 测量噪声协方差矩阵 R
R_noise = [std_meas^2, 0;
           0,          std_meas^2];

% 波门与航迹管理参数
gateThreshold = 10;     % 波门阈值（基于马氏距离）
maxLostFrames = 5;      % 最大允许连续漏检次数（超过则判定目标消失）

%% 3. 核心滤波与数据关联引擎
% 定义航迹结构体
targetList = struct('trackID', {}, 'stateVec', {}, 'CovMat', {}, ...
                'smoothPath', {}, 'measPath', {}, 'lostCount', {});
newTrackID = 1;

for k = 1:totalFrames
    currentFrameData = validMeasurements{k};
    Z_obs = currentFrameData(:, [1, 3]); % 提取本次扫描的位置 [x, y]
    obsNum = size(Z_obs, 1);
    trackNum = length(targetList);

    % --- 阶段A：状态预测 (Time Update) ---
    predStateList = cell(trackNum, 1);
    predCovList   = cell(trackNum, 1);
    for idx = 1:trackNum
        predStateList{idx} = TransMat * targetList(idx).stateVec;
        predCovList{idx}   = TransMat * targetList(idx).CovMat * TransMat' + Q_noise;
    end

    % --- 阶段B：数据关联 (最近邻 NNDA) ---
    isObsUsed = false(obsNum, 1);
    matchedPairs = nan(1, trackNum); 

    for tIdx = 1:trackNum
        x_predict = predStateList{tIdx};
        P_predict = predCovList{tIdx};
        z_predict = ObsMat * x_predict;
        
        minDist = inf;
        bestObsIdx = -1;

        % 寻找波门内马氏距离最小的量测点
        for oIdx = 1:obsNum
            if isObsUsed(oIdx), continue; end
            innovation = Z_obs(oIdx, :)' - z_predict;
            S_cov = ObsMat * P_predict * ObsMat' + R_noise; % 新息协方差
            mahalanobisDist = innovation' / S_cov * innovation;
            
            if mahalanobisDist < gateThreshold && mahalanobisDist < minDist
                minDist = mahalanobisDist;
                bestObsIdx = oIdx;
            end
        end
        if bestObsIdx > 0
            matchedPairs(tIdx) = bestObsIdx;
            isObsUsed(bestObsIdx) = true;
        end
    end

    % --- 阶段C：状态更新与修正 (Measurement Update) ---
    for tIdx = 1:trackNum
        x_predict = predStateList{tIdx};
        P_predict = predCovList{tIdx};

        if ~isnan(matchedPairs(tIdx))
            % 关联成功，计算卡尔曼增益并修正
            real_z = Z_obs(matchedPairs(tIdx), :)';
            S_cov = ObsMat * P_predict * ObsMat' + R_noise;
            KalmanGain = P_predict * ObsMat' / S_cov;
            
            x_update = x_predict + KalmanGain * (real_z - ObsMat * x_predict);
            P_update = (eye(4) - KalmanGain * ObsMat) * P_predict;
            
            targetList(tIdx).stateVec = x_update;
            targetList(tIdx).CovMat = P_update;
            targetList(tIdx).lostCount = 0;
            targetList(tIdx).smoothPath(end+1, :) = [x_update(1), x_update(3)];
            targetList(tIdx).measPath(end+1, :) = real_z';
        else
            % 发生漏检，依靠预测值维持航迹
            targetList(tIdx).stateVec = x_predict;
            targetList(tIdx).CovMat = P_predict;
            targetList(tIdx).lostCount = targetList(tIdx).lostCount + 1;
            targetList(tIdx).smoothPath(end+1, :) = (ObsMat * x_predict)';
            targetList(tIdx).measPath(end+1, :) = [NaN, NaN];
        end
    end

    % --- 阶段D：航迹起始与消亡管理 ---
    % 未被关联的观测点作为新的航迹起点
    for oIdx = 1:obsNum
        if ~isObsUsed(oIdx)
            init_x = Z_obs(oIdx, 1); init_y = Z_obs(oIdx, 2);
            % 估算初始速度
            if size(currentFrameData, 2) >= 4
                init_vx = currentFrameData(oIdx, 2); init_vy = currentFrameData(oIdx, 4);
            else
                init_vx = 0; init_vy = 0;
            end
            
            targetList(end+1).trackID = newTrackID;
            targetList(end).stateVec = [init_x; init_vx; init_y; init_vy];
            targetList(end).CovMat = diag([std_meas^2, 100, std_meas^2, 100]);
            targetList(end).smoothPath = [init_x, init_y];
            targetList(end).measPath = [init_x, init_y];
            targetList(end).lostCount = 0;
            newTrackID = newTrackID + 1;
        end
    end
    % 删除失联过久的航迹
    targetList = targetList([targetList.lostCount] < maxLostFrames);
end

%% 4. 跟踪结果全景可视化
figure('Name', '地波雷达多目标跟踪全景图', 'Position', [150, 150, 800, 600]);
hold on; grid on; box on;
colorMap = lines(length(targetList));

% 优化过滤：只绘制稳定跟踪长度超过 10 的有效航迹（过滤噪点）
validTrackIndices = find(arrayfun(@(x) size(x.smoothPath,1) > 10, targetList));

for idx = 1:length(validTrackIndices)
    tIdx = validTrackIndices(idx);
    tColor = colorMap(mod(idx, size(colorMap,1))+1, :);
    
    sPath = targetList(tIdx).smoothPath;
    mPath = targetList(tIdx).measPath;
    validPts = ~any(isnan(mPath), 2);
    
    % 绘制平滑轨迹
    plot(sPath(:,1), sPath(:,2), '-', 'Color', tColor, 'LineWidth', 2, ...
        'DisplayName', ['目标 ID: ', num2str(targetList(tIdx).trackID)]);
    % 绘制散点量测
    plot(mPath(validPts,1), mPath(validPts,2), '.', 'Color', tColor, ...
        'MarkerSize', 8, 'HandleVisibility', 'off');
end

xlabel('距离向坐标 X (km)'); ylabel('方位向坐标 Y (km)');
title('雷达多目标卡尔曼滤波跟踪结果 (滤除短噪点)');
legend('show', 'Location', 'best');
axis equal; set(gca, 'FontSize', 11, 'LineWidth', 1.2);