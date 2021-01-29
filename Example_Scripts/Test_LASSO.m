%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script compares SAGA, SVRG, SARAH, and SARGE on several LASSO 
% problems.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc
%%
strA = {'australian_label.mat', 'mushrooms_label.mat','phishing_label.mat','ijcnn1_label.mat'};%, 'rcv1_label.mat'};
strB = {'australian_sample.mat', 'mushrooms_sample.mat','phishing_sample.mat','ijcnn1_sample.mat'};%, 'rcv1_sample.mat'};

strF = {'australian', 'mushrooms','phishing','ijcnn1'};%,'rcv1'};

for j = 1:length(strF)
    
close all

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
for j=1:size(h,2)
    h(:,j) = rescale(h(:,j), -1, 1);
    if mod(j,1e2)==0; itsprint(sprintf('      column %06d...', j), j); end
end
fprintf(sprintf('\nDONE!\n\n'));
%% Prep
[m, d] = size(h);

n = d + 1;
para.m = m;
para.n = n;

W      = [h, ones(m, 1)];
para.W = W;
y      = l;
para.y = y;

para.mu = 1/m;

b = zeros(m, 1);
for i=1:m
    Wi = para.W(i,:);
    b(i) = norm(Wi)^2 /4;
end
para.beta_fi = 1 / max(b);

L = 1/para.beta_fi;

para.tol = 1e-11; % stopping criterion
para.maxits = 4e3*m; % max # of iteration

GradF = @(x) grad_lasso(x, para.W, para.y)/m;
iGradF = @(x, i) igrad_lasso(x, i, para.W, para.y);
iGradFOpt = @(x, i) igrad_lasso_SAGA_Lin(x,i,W,y);

ProxJ = @(x, t) sign(x).*max(abs(x) - t,0);

ObjF = @(x) func_lasso(x,para.W,para.y,para.mu);

outputType = 'pdf';



%% Compute a high-precision solution using SAGA

para.name       = [filename '/' 'high_p_lasso_' filename '.mat'];

if exist(para.name) ~= 2

    para.tol = 1e-15;

    para.Obj      = 1;

    para.theta    = 1;

    para.objEvery   = 100;
    para.saveEvery  = 1e4*m;
    para.printEvery = 100;

    para.c_gamma = 1/5;

    [x, its, ek, fk, sk, gk] = func_SAGA_Lin(para, iGradFOpt, ObjF, ProxJ);

    fprintf('\n');
       
    para.tol = 1e-11;
    
    save(para.name,'gk','sk','ek','fk','x','para')
    
end

high_p = load(para.name);







%% SAGA

theta_list = [1 10];

para.Obj      = 1;

para.objEvery   = m;
para.saveEvery  = 1e3*m;
para.printEvery = m;

its_old = -1;
mult    = 5;

for i = 1:length(theta_list)
    
    para.theta   = theta_list(i);
    para.name    = [filename '/saga_lasso_best_step_' filename '_theta_' num2str(para.theta) '_obj.mat'];
   
    if exist(para.name) ~= 2
        % Find the best step size from the set 1/(mult*L), where mult is a
        % natural number.
        while mult <= 100

            para.c_gamma             = 1/mult;
            [x, its, ek, fk, sk, gk] = func_SAGA_Lin(para, iGradFOpt, ObjF, ProxJ);

            if its_old < 0 || (its_old > 0 && its <= its_old)

                    x_saga(1:length(x),i)     = x;
                    its_saga(1:length(its),i) = its;
                    ek_saga(1:length(ek),i)   = ek;
                    fk_saga(1:length(fk),i)   = fk;
                    sk_saga(1:length(sk),i)   = sk;
                    gk_saga(1:length(gk),i)   = gk;
                    stepsize_times_L(i)       = 1/mult;
                    
                    its_old                   = its;
                    mult                      = mult+1;

            else
                mult    = 5;
                its_old = -1;
                break
            end
            
        end
        
    else
        para_old = para;
        load(para.name)
        para = para_old;

        x_saga(1:length(x),i)     = x;
        its_saga(1:length(its),i) = its;
        ek_saga(1:length(ek),i)   = ek;
        fk_saga(1:length(fk),i)   = fk;
        sk_saga(1:length(sk),i)   = sk;
        gk_saga(1:length(gk),i)   = gk;

    end
            
end


fprintf('\n');





%% SVRG

epoch_svrg = 2*m;
para.P     = epoch_svrg;

theta_list = [1 1.2];

para.Obj      = 1;

para.objEvery   = m;
para.saveEvery  = 1e3*m;
para.printEvery = m;
para.c_gamma = 1/5;

its_old = -1;
mult    = 5;

for i = 1:length(theta_list)
    
    para.c_gamma = 1/mult;
    para.theta   = theta_list(i);
    para.name       = [filename '/svrg_lasso_best_step_' filename '_theta_' num2str(para.theta) '_obj.mat'];
   
    if exist(para.name) ~= 2
    
        while mult <= 100
            
            para.c_gamma              = 1/mult;
            [x, its, ek, fk, sk, gk] = func_SVRG(para, GradF, iGradF, ObjF, ProxJ);

            if its_old < 0 || (its_old > 0 && its <= its_old)

                    x_svrg(1:length(x),i)     = x;
                    its_svrg(1:length(its),i) = its;
                    ek_svrg(1:length(ek),i)   = ek;
                    fk_svrg(1:length(fk),i)   = fk;
                    sk_svrg(1:length(sk),i)   = sk;
                    gk_svrg(1:length(gk),i)   = gk;
                    stepsize_times_L(i)       = 1/mult;
                    
                    its_old                   = its;
                    mult                      = mult+1;

            else
                mult    = 5;
                its_old = -1;
                break
            end
            
        end
        
    else
        para_old = para;
        load(para.name)
        para = para_old;

        x_svrg(1:length(x),i)     = x;
        its_svrg(1:length(its),i) = its;
        ek_svrg(1:length(ek),i)   = ek;
        fk_svrg(1:length(fk),i)   = fk;
        sk_svrg(1:length(sk),i)   = sk;
        gk_svrg(1:length(gk),i)   = gk;

    end
            
end


fprintf('\n');




%% SARAH

epoch_sarah = 2*m;
para.q     = 1/epoch_sarah;


para.Obj      = 1;
para.tol      = 1e-14;

para.objEvery   = m;
para.saveEvery  = 1e3*m;
para.printEvery = m;

mult         = 5;
its_old      = -1;

para.name       = [filename '/sarah_lasso_best_step_' filename '_obj.mat'];


if exist(para.name) ~= 2

while mult <= 100
    
    para.c_gamma = 1/mult;
    [x, its, ek, fk, sk, gk] = func_rSARAH(para, GradF, iGradF, ObjF, ProxJ);

    if its_old > 0
        if its <= its_old

            x_sarah    = x;
            its_sarah  = its;
            ek_sarah   = ek;
            fk_sarah   = fk;
            sk_sarah   = sk;
            gk_sarah   = gk;
            stepsize_times_L(i) = 1/mult;

        else
            break
        end
    else

        x_sarah    = x;
        its_sarah  = its;
        ek_sarah   = ek;
        fk_sarah   = fk;
        sk_sarah   = sk;
        gk_sarah   = gk;
        stepsize_times_L(i) = 1/mult;

    end

    mult    = mult+1;
    its_old = its;

end


else
    para_old = para;
    load(para.name)
    para = para_old;
    
    x_sarah(1:length(x),1)     = x;
    ek_sarah(1:length(ek),1)   = ek;
    fk_sarah(1:length(fk),1)   = fk;
    sk_sarah(1:length(sk),1)   = sk;
    gk_sarah(1:length(gk),1)   = gk;
end

fprintf('\n');


%% SARGE

para.Obj      = 1;
para.tol      = 1e-11;

para.objEvery   = m;
para.saveEvery  = 1e3*m;
para.printEvery = m;

mult         = 5;
its_old      = -1;

para.name       = [filename '/sarge_lasso_best_step_' filename '_obj.mat'];

if exist(para.name) ~= 2

while mult <= 100
    para.c_gamma = 1/mult;

    [x, its, ek, fk, sk, gk] = func_SARGE_Lin(para, iGradFOpt, ObjF, ProxJ);

    if its_old > 0
        if its <= its_old

            x_sarge    = x;
            its_sarge  = its;
            ek_sarge   = ek;
            fk_sarge   = fk;
            sk_sarge   = sk;
            gk_sarge   = gk;
            stepsize_times_L(i) = 1/mult;

        else
            break
        end
    else

        x_sarge    = x;
        its_sarge  = its;
        ek_sarge   = ek;
        fk_sarge   = fk;
        sk_sarge   = sk;
        gk_sarge   = gk;
        stepsize_times_L(i) = 1/mult;

    end

    mult    = mult+1;
    its_old = its;

end


else
    para_old = para;
    load(para.name)
    para = para_old;
    
    x_sarge(1:length(x),1)     = x;
    its_sarge(1:length(its),1) = its;
    ek_sarge(1:length(ek),1)   = ek;
    fk_sarge(1:length(fk),1)   = fk;
    sk_sarge(1:length(sk),1)   = sk;
    gk_sarge(1:length(gk),1)   = gk;
end


fprintf('\n');





%% Compare Objective Plot

exact = high_p.fk(end);

x_svrg  = 1:(2+epoch_svrg/m):length(fk_svrg)*(2+epoch_svrg/m);
x_sarah = 1:(2+epoch_sarah/m):length(fk_sarah)*(2+epoch_sarah/m);

linewidth = 1.5;

axesFontSize = 8;
labelFontSize = 8;
legendFontSize = 10;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(100), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.0 -0.025 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.4]);

p1 = semilogy(fk_saga(:,1) - exact, '-k', 'LineWidth',linewidth);
hold on
p2 = semilogy(fk_saga(:,2) - exact, '-', 'LineWidth',linewidth);
p3 = semilogy(fk_svrg(:,1) - exact, '-r', 'LineWidth',linewidth);
p4 = semilogy(fk_sarah - exact, '-b', 'LineWidth',linewidth);
p5 = semilogy(fk_sarge - exact, '-m', 'LineWidth',linewidth);


grid on;
ax = gca;
ax.GridLineStyle = '--';

ylim([1e-15 1]);

ylabel({'$F(x_k) - F(x^*)$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-0.0mm}';'$k/n$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');

lg = legend([p1, p2, p3, p4, p5],'SAGA','B-SAGA, $\theta = 10$','SVRG','SARAH','SARGE');
set(lg,'Location', 'Best');
set(lg,'FontSize', legendFontSize);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

% save figure
epsname = sprintf('%s/lasso_comparison_%s_obj.%s', filename, filename, outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end

clear x1_theta its1_theta ek1_theta fk1_theta sk1_theta gk1_theta stepsize_times_L

end