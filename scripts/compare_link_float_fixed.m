%% compare_link_float_fixed.m
% 对比无 DPD / 浮点 DPD / 定点 DPD 的 EVM & ACLR（使用 out 结构体）

%% 0. 从 base workspace 取仿真输出 out & 脚本输入 tx_bb
out   = evalin('base','out');     % Simulink 单一输出
tx_bb = evalin('base','tx_bb(:);');  % 你脚本里生成的基带输入

%% 1. 从 out 中取信号，转成列向量
x_ref_sim       = out.x_ref_sim(:);        % Simulink 输入
y_no_dpd_sim    = out.y_no_dpd_sim(:);     % 无 DPD 输出
y_dpd_float_sim = out.y_dpd_float_sim(:);  % 浮点 DPD 输出
y_dpd_fixed_sim = out.y_dpd_fixed_sim(:);  % 定点 DPD 输出

%% 2. 对齐长度（取最短）
Nmin = min([length(tx_bb), length(x_ref_sim), ...
            length(y_no_dpd_sim), length(y_dpd_float_sim), length(y_dpd_fixed_sim)]);
tx_bb          = tx_bb(1:Nmin);
x_ref_sim      = x_ref_sim(1:Nmin);
y_no_dpd_sim   = y_no_dpd_sim(1:Nmin);
y_dpd_float_sim= y_dpd_float_sim(1:Nmin);
y_dpd_fixed_sim= y_dpd_fixed_sim(1:Nmin);

fprintf('有效对比长度 Nmin = %d\n', Nmin);

%% 3. 检查：脚本输入 vs Simulink 输入是否一致
diff_x = tx_bb - x_ref_sim;
fprintf('tx_bb vs x_ref_sim 最大差值 = %.3e\n', max(abs(diff_x)));

% 后面算 EVM/ACLR 时，用哪个参考？
% -> 建议直接用 x_ref_sim，保证和 Simulink 内链路绝对一致。
x_ref = x_ref_sim;

%% 4. 加载 OFDM 参数
run('initNR_OFDM.m');  % 确保 NR.Fs 在当前 workspace
Fs      = NR.Fs;
BW_main = 98e6;

%% 5. 无 DPD：EVM / ACLR
g_no   = (x_ref' * y_no_dpd_sim) / (x_ref' * x_ref);
y_lin_no = g_no * x_ref;
err_no   = y_no_dpd_sim - y_lin_no;
EVM_no   = sqrt(mean(abs(err_no).^2)/mean(abs(y_lin_no).^2))*100;
ACLR_no  = myACLR_fft(y_no_dpd_sim, Fs, BW_main);

%% 6. 浮点 NN-DPD：EVM / ACLR
g_f    = (x_ref' * y_dpd_float_sim) / (x_ref' * x_ref);
y_lin_f = g_f * x_ref;
err_f   = y_dpd_float_sim - y_lin_f;
EVM_f   = sqrt(mean(abs(err_f).^2)/mean(abs(y_lin_f).^2))*100;
ACLR_f  = myACLR_fft(y_dpd_float_sim, Fs, BW_main);

%% 7. 定点 NN-DPD：EVM / ACLR
g_fx     = (x_ref' * y_dpd_fixed_sim) / (x_ref' * x_ref);
y_lin_fx = g_fx * x_ref;
err_fx   = y_dpd_fixed_sim - y_lin_fx;
EVM_fx   = sqrt(mean(abs(err_fx).^2)/mean(abs(y_lin_fx).^2))*100;
ACLR_fx  = myACLR_fft(y_dpd_fixed_sim, Fs, BW_main);

%% 8. 打印结果
fprintf('\n=== 完整链路对比（Simulink） ===\n');
fprintf('无 DPD：        EVM = %.2f %%, ACLR = %.2f dB\n', EVM_no,  ACLR_no);
fprintf('浮点 NN-DPD：   EVM = %.2f %%, ACLR = %.2f dB\n', EVM_f,   ACLR_f);
fprintf('定点 NN-DPD：   EVM = %.2f %%, ACLR = %.2f dB\n', EVM_fx,  ACLR_fx);