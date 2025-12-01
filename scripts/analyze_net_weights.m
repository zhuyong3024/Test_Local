%% analyze_net_weights.m
% 目的：分析 netPost 中各全连接层权重/偏置的数值范围
%      为后续权重量化 (fixed-point NN) 提供依据

clear; clc;

%% 1. 加载训练好的网络
S = load('../nn/net_dpd_post.mat', 'netPost', 'K', 'NR');
netPost = S.netPost;

% 打印网络结构，确认层的名字
disp(netPost.Layers);

%% 2. 找出全连接层并统计权重/偏置范围
layers = netPost.Layers;

fprintf('\n=== 全连接层权重/偏置范围统计 ===\n');

for i = 1:numel(layers)
    L = layers(i);
    if isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
        name = L.Name;
        W = L.Weights;   % size: [numNeurons × numInputs]
        b = L.Bias;      % size: [numNeurons × 1]

        W_abs = abs(W(:));
        b_abs = abs(b(:));

        fprintf('\n[%s]\n', name);
        fprintf('  Weights:  max|W| = %.4f,  mean|W| = %.4f,  99.9%%|W| = %.4f\n', ...
            max(W_abs), mean(W_abs), prctile(W_abs, 99.9));
        fprintf('  Biases :  max|b| = %.4f,  mean|b| = %.4f,  99.9%%|b| = %.4f\n', ...
            max(b_abs), mean(b_abs), prctile(b_abs, 99.9));
    end
end

%% 3. 可选：画权重直方图（看分布）
figure;
plotIdx = 1;
for i = 1:numel(layers)
    L = layers(i);
    if isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
        W_abs = abs(L.Weights(:));
        subplot(2,2,plotIdx);
        histogram(W_abs, 100);
        title(sprintf('Layer %s |W|', L.Name));
        xlabel('|W|'); ylabel('Count');
        plotIdx = plotIdx + 1;
    end
end
sgtitle('NN-DPD 各全连接层权重幅度直方图');