%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script compares SAGA, SVRG, SARAH, and SARGE on several 
% non-negative PCA problems.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc
%%
strA = {'australian_label.mat', 'mushrooms_label.mat','phishing_label.mat','ijcnn1_label.mat'};
strB = {'australian_sample.mat', 'mushrooms_sample.mat','phishing_sample.mat','ijcnn1_sample.mat'};

strF = {'australian', 'mushrooms','phishing','ijcnn1'};

for j = 1:length(strF)

i_file = j;
%% load and scale data
class_name = strA{i_file};
feature_name = strB{i_file};

filename = strF{i_file};

mkdir(filename);

load(['../data/', class_name]);
load(['../data/', feature_name]);

h = full(h);

% rescale the data
fprintf(sprintf('rescale data...\n'));
itsprint(sprintf('      column %06d...', 1), 1);
for t=1:size(h,2)
    h(:,t) = rescale(h(:,t), -1, 1);
    if mod(t,1e2)==0; itsprint(sprintf('      column %06d...', j), j); end
end
fprintf(sprintf('\nDONE!\n\n'));
%% Prep
[m, d] = size(h);

n = d + 1;
para.m = m;
para.n = n;
para.mu = 1;

W      = [h, ones(m, 1)];
para.W = W;
y      = zeros(m,1);
para.y = y;

b = zeros(m, 1);
for i=1:m
    Wi = para.W(i,:);
    b(i) = norm(Wi)^2/4;
end
para.beta_fi = 1/max(b);

L = 1/para.beta_fi;

para.tol = 1e-11; % stopping criterion
para.maxits = 1e4*m; % max # of iteration

% Proximal operator for non-negative PCA
ProxJ = @(x, t) (x./max(1,norm(x,'fro'))).*sign(x);

GradF     = @(x) -grad_l2(x, W, y)/m;
iGradFOpt = @(x, i) -igrad_l2_SAGA_Lin(x,i,W,y);
iGradF    = @(x, i) -igrad_l2(x,i,W,y);

ObjF = @(x) func_nnPCA(x, W, y, 0);

outputType = 'pdf';

% Same initial point for all tests
para.x0 = randn(n,1); % always start from non-zero point
para.x0 = ProxJ(para.x0,1); % project onto constraint set



%% Compute a high-precision solution using PGD

para.name       = [filename '/' 'high_p_nnPCA_' filename '.mat'];

if exist(para.name) ~= 2

    para.P   = 1;
    para.tol = 1e-13;
    para.Obj      = 1;
    para.theta    = 1;
    para.objEvery   = 1e4;
    para.saveEvery  = 1e4*m;
    para.printEvery = 1e4;

    para.c_gamma = 1/(5*m);

    % para.x0 = randn(n,1); % always start from non-zero point
    % para.x0 = ProxJ(para.x0,1); % project onto constraint set

    [x_best, its_best, ek_best, fk_best, sk_best, gk_best] = func_SVRG(para, GradF, iGradF, ObjF, ProxJ);
    save(para.name,'gk_best','sk_best','ek_best','fk_best','x_best','para')

    [x_temp, its_temp, ek_temp, fk_temp, sk_temp, gk_temp] = func_SVRG(para, GradF, iGradF, ObjF, ProxJ);

    fprintf('\n');

    if min(fk_temp) < min(fk_best)

        x_best = x_temp;
        its_best = its_temp;
        ek_best = ek_temp;
        fk_best = fk_temp;
        sk_best = sk_temp;
        gk_best = gk_temp;

        save(para.name,'gk_best','sk_best','ek_best','fk_best','x_best','para')

    end

end

high_p = load(para.name);




%% SAGA

para.tol = 1e-14;

theta_list = [0.1 1];

para.Obj      = 1;

para.objEvery   = m;
para.saveEvery  = 1e3*m;
para.printEvery = m;

para.c_gamma = 1/(5*m);
para.maxits = 5e2*m; % max # of iteration

for i = 1:length(theta_list)
    para.theta    = theta_list(i);
    
    para.name       = [filename '/saga_nnPCA_' filename '_theta_' num2str(para.theta) '_obj.mat'];
    
    if exist(para.name) ~= 2
        [x, its, ek, fk, mean_fk, sk, gk] = func_SAGA_Lin_noncon(para, iGradFOpt, ObjF, ProxJ);
    else
        para_old = para;
        load(para.name)
        para = para_old;
    end
    
    x_saga(1:length(x),i)     = x;
    ek_saga(1:length(ek),i)   = ek;
    fk_saga(1:length(mean_fk),i)   = mean_fk;
    sk_saga(1:length(sk),i)   = sk;
    gk_saga(1:length(gk),i)   = gk;
    
end

fprintf('\n');







%% SVRG

para.tol = 1e-14;

theta_list = [0.1 1];

para.Obj      = 1;

para.P        = 2*m;

para.objEvery   = m;
para.saveEvery  = 1e3*m;
para.printEvery = m;

para.c_gamma = 1/(5*m);
para.maxits = 5e2*m; % max # of iteration

for i = 1:length(theta_list)
    para.theta    = theta_list(i);
    
    para.name       = [filename '/svrg_nnPCA_' filename '_theta_' num2str(para.theta) '_obj.mat'];
    
    if exist(para.name) ~= 2
        [x, its, ek, fk, mean_fk, sk, gk] = func_SVRG_noncon(para, GradF, iGradF, ObjF, ProxJ);
    else
        para_old = para;
        load(para.name)
        para = para_old;
    end
    
    x_svrg(1:length(x),i)     = x;
    ek_svrg(1:length(ek),i)   = ek;
    fk_svrg(1:length(mean_fk),i) = mean_fk;
    sk_svrg(1:length(sk),i)   = sk;
    gk_svrg(1:length(gk),i)   = gk;
    
end

fprintf('\n');







%% SARAH

para.tol = 1e-14;

para.Obj      = 1;

para.q        = 1/(2*m);

para.objEvery   = m;
para.saveEvery  = 1e3*m;
para.printEvery = m;

para.c_gamma = 1/(5*m);
para.maxits = 5e2*m; % max # of iteration
    
para.name       = [filename '/sarah_nnPCA_' filename '_obj.mat'];

if exist(para.name) ~= 2
    [x, its, ek, fk, mean_fk, sk, gk] = func_rSARAH_noncon(para, GradF, iGradF, ObjF, ProxJ);
else
    para_old = para;
    load(para.name)
    para = para_old;
end

x_sarah     = x;
ek_sarah    = ek;
fk_sarah    = mean_fk;
sk_sarah    = sk;
gk_sarah    = gk;
    

fprintf('\n');








%% SARGE

para.tol = 1e-12;

para.Obj      = 1;

para.objEvery   = m;
para.saveEvery  = 1e3*m;
para.printEvery = m;

para.c_gamma = 1/(5*m);
para.maxits = 5e2*m; % max # of iteration
    
para.name       = [filename '/sarge_nnPCA_' filename '_obj.mat'];

if exist(para.name) ~= 2
    [x, its, ek, fk, mean_fk, sk, gk] = func_SARGE_Lin_noncon(para, iGradFOpt, ObjF, ProxJ);
else
    para_old = para;
    load(para.name)
    para = para_old;
end

x_sarge     = x;
ek_sarge    = ek;
fk_sarge    = mean_fk;
sk_sarge    = sk;
gk_sarge    = gk;
    

fprintf('\n');










%% Compare Objective Plot

exact = high_p.fk_best(end);

linewidth = 1.5;

axesFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(100), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.0 -0.025 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.4]);

p1 = semilogy(fk_saga(:,2) - exact, '-k', 'LineWidth',linewidth);
hold on
p2 = semilogy(fk_saga(:,1) - exact, '-', 'LineWidth',linewidth);
p3 = semilogy(fk_svrg(:,2) - exact, '--r', 'LineWidth',linewidth);
p4 = semilogy(fk_svrg(:,1) - exact, '--', 'LineWidth',linewidth);
p5 = semilogy(fk_sarah - exact, 'b', 'LineWidth',linewidth);
p6 = semilogy(fk_sarge - exact, '-m', 'LineWidth',linewidth);

grid on;
ax = gca;
ax.GridLineStyle = '--';

ylabel({'$F(x_k) - F(x^*)$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-0.0mm}';'$k/n$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');

lg = legend([p1, p2, p3, p4, p5, p6],'SAGA','B-SAGA, $\theta = 0.1$','SVRG','B-SVRG, $\theta = 0.1$','SARAH','SARGE');
set(lg,'Location', 'Best');
set(lg,'FontSize', 10);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

epsname = sprintf('%s/nnPCA_comparison_%s_obj.%s', filename, filename, outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end

clear x1_theta its1_theta ek1_theta fk1_theta sk1_theta gk1_theta stepsize_times_L

end
