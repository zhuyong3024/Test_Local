function xq = quant_uniform(x, B, xmax)
% quant_uniform  均匀量化 + 饱和，模拟 B 位有符号定点（双边饱和）
%
% 输入：
%   x    : 输入信号（double，实或复）
%   B    : 总位宽（含符号位），例如 12, 14, 16
%   xmax : 量化的正饱和幅度（abs(x) 超过此值会被饱和到 ±xmax）
%
% 输出：
%   xq   : 量化后的信号（double，近似于 B 位定点）

% 允许的最大码值（有符号，1 位符号 + (B-1) 位幅度）
L = 2^(B-1) - 1;

% 步长
delta = xmax / L;

% 对实部和虚部分别量化（支持复数）
x_re = real(x);
x_im = imag(x);

% 饱和
x_re = max(min(x_re, xmax), -xmax);
x_im = max(min(x_im, xmax), -xmax);

% 量化到最近的整数码字
x_re_q = round(x_re / delta) * delta;
x_im_q = round(x_im / delta) * delta;

xq = complex(x_re_q, x_im_q);
end