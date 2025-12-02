%% compare_link_float_fixed.m
% 对比无 DPD / 浮点 DPD / 定点 DPD 的 EVM & ACLR（使用 out 结构体）

out   = evalin('base','out');
tx_bb = evalin('base','tx_bb(:);');

x_ref_sim       = out.x_ref_sim(:);
y_no_dpd_sim    = out.y_no_dpd_sim(:);
y_dpd_float_sim = out.y_dpd_float_sim(:);
y_dpd_fixed_sim = out.y_dpd_fixed_sim(:);

Nmin = min([length(tx_bb), length(x_ref_sim), ...
            length(y_no_dpd_sim), length(y_dpd_float_sim), length(y_dpd_fixed_sim)]);
tx_bb          = tx_bb(1:Nmin);
x_ref_sim      = x_ref_sim(1:Nmin);
y_no_dpd_sim   = y_no_dpd_sim(1:Nmin);
y_dpd_float_sim= y_dpd_float_sim(1:Nmin);
y_dpd_fixed_sim= y_dpd_fixed_sim(1:Nmin);

fprintf('有效对比长度 Nmin = %d\n', Nmin);
fprintf('tx_bb vs x_ref_sim 最大差值 = %.3e\n', max(abs(tx_bb-x_ref_sim)));

% 用 x_ref_sim 做参考
x_ref = x_ref_sim;

run('initNR_OFDM.m');
Fs      = NR.Fs;
BW_main = 98e6;

% 无 DPD
g_no   = (x_ref' * y_no_dpd_sim) / (x_ref' * x_ref);
y_lin_no = g_no * x_ref;
err_no   = y_no_dpd_sim - y_lin_no;
EVM_no   = sqrt(mean(abs(err_no).^2)/mean(abs(y_lin_no).^2))*100;
ACLR_no  = myACLR_fft(y_no_dpd_sim, Fs, BW_main);

% 浮点 DPD
g_f    = (x_ref' * y_dpd_float_sim) / (x_ref' * x_ref);
y_lin_f = g_f * x_ref;
err_f   = y_dpd_float_sim - y_lin_f;
EVM_f   = sqrt(mean(abs(err_f).^2)/mean(abs(y_lin_f).^2))*100;
ACLR_f  = myACLR_fft(y_dpd_float_sim, Fs, BW_main);

% 定点 DPD
g_fx     = (x_ref' * y_dpd_fixed_sim) / (x_ref' * x_ref);
y_lin_fx = g_fx * x_ref;
err_fx   = y_dpd_fixed_sim - y_lin_fx;
EVM_fx   = sqrt(mean(abs(err_fx).^2)/mean(abs(y_lin_fx).^2))*100;
ACLR_fx  = myACLR_fft(y_dpd_fixed_sim, Fs, BW_main);

% 浮点 vs 定点输出间 EVM
N2   = min(length(y_dpd_float_sim), length(y_dpd_fixed_sim));
yf   = y_dpd_float_sim(1:N2);
yfx  = y_dpd_fixed_sim(1:N2);
g_ffx   = (yf' * yfx) / (yf' * yf);
y_lin_ffx = g_ffx * yf;
err_ffx   = yfx - y_lin_ffx;
EVM_ffx   = sqrt(mean(abs(err_ffx).^2)/mean(abs(y_lin_ffx).^2))*100;

fprintf('\n=== 完整链路对比（Simulink，优化定点后） ===\n');
fprintf('无 DPD：        EVM = %.2f %%, ACLR = %.2f dB\n', EVM_no,  ACLR_no);
fprintf('浮点 NN-DPD：   EVM = %.2f %%, ACLR = %.2f dB\n', EVM_f,   ACLR_f);
fprintf('定点 NN-DPD：   EVM = %.2f %%, ACLR = %.2f dB\n', EVM_fx,  ACLR_fx);
fprintf('浮点输出 vs 定点输出 EVM = %.2f %%\n', EVM_ffx);
