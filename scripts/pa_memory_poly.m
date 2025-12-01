function y = pa_memory_poly(x)
% pa_memory_poly  简单记忆多项式功放模型
%
%   y = pa_memory_poly(x)
%
% 模型形式：
%   y[n] = sum_{k=0}^{K-1} sum_{p=1,3,5} a_{p,k} * x[n-k] * |x[n-k]|^{p-1}
%
% 这里系数是人工设定的，只是为了制造"明显非线性 + 一点记忆效应"，
% 后面你可以换成用测量数据拟合出来的系数。

x = x(:);  % 保证是列向量
N = length(x);

% 参数：最大阶数 p=5，记忆深度 K=3（可按需要调）
K = 3;   % 记忆 tap 数
% 系数矩阵 a(pIndex, kIndex)，pIndex=1→1阶, 2→3阶, 3→5阶
a = zeros(3, K);

% 1 阶（近似线性增益 + 微小记忆）
a(1,:) = [1.0, -0.05, 0.02];
% 3 阶（主非线性）
a(2,:) = [0.8, -0.04, 0.015];
% 5 阶（高阶失真）
a(3,:) = [0.3, -0.02, 0.010];

% 输出缓存
y = complex(zeros(N,1));

for n = 1:N
    acc = 0;
    for k = 0:(K-1)
        idx = n - k;
        if idx < 1
            continue;
        end
        xnk = x(idx);
        r   = abs(xnk);
        % 1,3,5 阶项
        acc = acc ...
            + a(1,k+1) * xnk ...
            + a(2,k+1) * xnk * (r^2) ...
            + a(3,k+1) * xnk * (r^4);
    end
    y(n) = acc;
end

end