%% compare_DPD_poly.m
% 比较：无 DPD vs 多项式 DPD 的 EVM 和 ACLR

clear; clc;

%% 1. 重新生成基带信号和 PA 输出（无 DPD）
NR    = initNR_OFDM();
tx_bb = genOFDMFrame(NR);

% 脚本版 PA 输出（无 DPD）
y_pa_noDPD = pa_memory_poly(tx_bb);

%% 2. 运行 Simulink 模型，得到"有 DPD"的输出
Ts = NR.Ts;
Fs = NR.Fs;

t     = (0:length(tx_bb)-1).' * Ts;
tx_ts = timeseries(tx_bb, t);

% 把变量放进 base workspace 供 Simulink 使用
assignin('base','NR',NR);
assignin('base','tx_bb',tx_bb);
assignin('base','Ts',Ts);
assignin('base','Fs',Fs);
assignin('base','tx_ts',tx_ts);

% 运行 Simulink 模型 dpd_top，并显式接收仿真输出
simOut = sim('dpd_top', ...
    'StopTime', num2str(length(tx_bb)*Ts));

% 从仿真输出结构体中获取 To Workspace 记录的信号
% 注意：前提是 To Workspace 的 Variable name = 'y_pa_dpd_sim'
y_pa_dpd = simOut.y_pa_dpd_sim;

%% 3. 对齐长度
Nmin = min([length(tx_bb), length(y_pa_noDPD), length(y_pa_dpd)]);
tx_ref       = tx_bb(1:Nmin);
y_pa_noDPD   = y_pa_noDPD(1:Nmin);
y_pa_dpd     = y_pa_dpd(1:Nmin);

%% 4. 计算波形 EVM-like（去掉线性增益）
% 无 DPD
g0   = (tx_ref' * y_pa_noDPD) / (tx_ref' * tx_ref);
y0_l = g0 * tx_ref;
err0 = y_pa_noDPD - y0_l;
EVM0 = sqrt(mean(abs(err0).^2) / mean(abs(y0_l).^2)) * 100;

% 有 DPD
g1   = (tx_ref' * y_pa_dpd) / (tx_ref' * tx_ref);
y1_l = g1 * tx_ref;
err1 = y_pa_dpd - y1_l;
EVM1 = sqrt(mean(abs(err1).^2) / mean(abs(y1_l).^2)) * 100;

%% 5. 计算 ACLR（用之前的 myACLR_fft，主带 98 MHz）
BW_main = 98e6;

ACLR0 = myACLR_fft(y_pa_noDPD, Fs, BW_main);
ACLR1 = myACLR_fft(y_pa_dpd,   Fs, BW_main);

%% 6. 打印结果
fprintf('无 DPD：EVM = %.2f %%, ACLR = %.2f dB\n', EVM0, ACLR0);
fprintf('有 DPD：EVM = %.2f %%, ACLR = %.2f dB\n', EVM1, ACLR1);