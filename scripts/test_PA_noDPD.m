%% test_PA_noDPD.m
% 目的：基于 MATLAB 端到端测试 OFDM → PA，查看失真和指标（无 DPD）

clear; clc;

%% 1. 初始化 OFDM 参数 + 生成基带信号
NR    = initNR_OFDM();
tx_bb = genOFDMFrame(NR);

fprintf('tx_bb length = %d samples\n', length(tx_bb));

%% 2. 通过功放模型
y_pa = pa_memory_poly(tx_bb);

%% 3. 对比输入输出的波形和频谱
figure;
subplot(2,1,1);
plot(real(tx_bb(1:200)));
grid on;
title('Input tx\_bb (real, first 200 samples)');

subplot(2,1,2);
plot(real(y_pa(1:200)));
grid on;
title('PA output y\_pa (real, first 200 samples)');

% 频谱对比
figure;
subplot(2,1,1);
pwelch(tx_bb, [], [], [], NR.Fs, 'centered');
title('Spectrum of input tx\_bb');

subplot(2,1,2);
pwelch(y_pa, [], [], [], NR.Fs, 'centered');
title('Spectrum of PA output y\_pa');

%% 4. 计算一个"波形 EVM-like 指标"（去掉线性增益）
% 用最小二乘估计一个最佳线性增益 g，使得 g*tx_bb 拟合 y_pa
g = (tx_bb' * y_pa) / (tx_bb' * tx_bb);
y_lin = g * tx_bb;

err = y_pa - y_lin;

EVM_wave = sqrt(mean(abs(err).^2) / mean(abs(y_lin).^2)) * 100;
fprintf('Waveform-based EVM-like = %.2f %%\n', EVM_wave);

%% 5. 计算 ACLR（主信道带宽 98 MHz，自定义 FFT 版本）
BW_main = 98e6;  % 主信道带宽 98 MHz（≈ Nsc * 30kHz）

acpr_dB = myACLR_fft(y_pa, NR.Fs, BW_main);
fprintf('ACLR (custom FFT, BWmain=98MHz) = %.2f dB\n', acpr_dB);