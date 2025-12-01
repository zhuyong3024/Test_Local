function y_pred = nn_dpd_predict(feat)
% nn_dpd_predict  外部 NN 预测封装函数（供 MATLAB Function 块以 extrinsic 方式调用）
%
% 输入：
%   feat : 1×(2K) 的特征行向量（double）
% 输出：
%   y_pred : 1×2 的 [Re, Im] 行向量（double）

persistent netLoaded net

if isempty(netLoaded)
    data = load('../nn/net_dpd_post.mat', 'netPost');
    net  = data.netPost;
    netLoaded = true;
    fprintf('nn_dpd_predict: 网络已加载\n');
end

% 确保是 double 行向量
feat = double(feat);

yp = predict(net, feat);   % Deep Learning Toolbox 前向推理
y_pred = double(yp);       % 转成 double，尺寸为 1×2

end