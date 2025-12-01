function xq = quant_uniform_real(x, B, xmax)
% quant_uniform_real  实数均匀量化 + 饱和，模拟 B 位有符号定点
%
% 输入：
%   x    : 输入信号（double，实数矩阵）
%   B    : 总位宽（含符号位），例如 6, 8, 10, 12
%   xmax : 量化的正饱和幅度（|x| > xmax 会被饱和到 ±xmax）
%
% 输出：
%   xq   : 量化后的信号（double，尺寸与 x 相同）
%
% 量化区间：[-xmax, xmax]
% 有符号定点：1 位符号 + (B-1) 位幅度，共 2^(B-1)-1 个正码值

% 最大正码值
L = 2^(B-1) - 1;

% 步长
delta = xmax / L;

% 饱和
x_sat = max(min(x, xmax), -xmax);

% 量化到最近的码字
xq = round(x_sat / delta) * delta;
end
