function [x, its, ek, fk, mean_fk, sk, tk, gk] = func_rSARAH_noncon(para, GradF ,iGradF, ObjF, ProxJ)
%
% Uses SARAH with an epoch length determined by a random variable to solve 
% the problem
%
%    min_x   1/n sum_{i=1}^n f_i(x) + mu * J(x)
%
% where f_i has a Lipschitz continuous gradient for all i. 
%
% Inputs:
%   para   - struct of parameters
%   GradF  - function handle mapping x to the gradient of 1/n sum_{i=1}^n f_i(x)
%   iGradF - function handle mapping (x,i) the gradient of f_i at x
%   ObjF   - function handle returning the objective value at x
%   ProxJ  - function handle returning the proximal operator of the
%                non-smooth term
%
% Outputs:
%   x   - minimiser
%   its - number of iterations
%   ek  - distance between iterations ||x_k - x_{k-1}||_2
%   fk  - objective value
%   mean_fk - objective value averaged over one epoch
%   sk  - number of non-zero entries of x (size of support)
%   tk  - time
%   gk  - step-size
%
% Parameters:
%     m          - number of functions in smooth component;
%     n          - length of vector x;
%     mu         - non-negative tuning parameter ( n^(-1/2) );
%     q          - determines epoch length: compute the full gradient with
%                  probability q every iteration (1/(2*m));
%     c_gamma    - step-size times Lipschitz constant of f_i (0.1);
%     beta_fi    - inverse of Lipschitz constant of f_i (1);
%     maxits     - maximum number of iterations (1000);
%     printEvery - number of iterations between printing (100);
%     saveEvery  - number of iterations between saves (100);
%     objEvery   - number of iterations between objective prints (100);
%     printObj   - print objective values? (True);
%     tol        - tolerance in distance between iterates (1e-4)
%     window     - because of nonconvexity, function values are averaged
%                   over an epoch (m).

% set problem dimensions
if isfield(para,'m') && isfield(para,'n')
    m = para.m;
    n = para.n;
else
    error('Must provide problem dimensions para.m and para.n')
end

% set parameters
mu      = setOpts(para,'mu',1/sqrt(m));
q       = setOpts(para,'q',1/(2*m));
c_gamma = setOpts(para,'c_gamma',0.1);
beta_fi = setOpts(para,'beta_fi',1);
maxits  = setOpts(para,'maxits',1000);
printEvery = setOpts(para,'printEvery',100);
saveEvery  = setOpts(para,'saveEvery',100);
printObj   = setOpts(para,'printObj',1);
objEvery  = setOpts(para,'objEvery',100);
b          = setOpts(para,'b',1);
tol        = setOpts(para,'tol',1e-4);
x0         = setOpts(para,'x0',randn(n,1));
window     = setOpts(para,'window',m);

% regulate batch size
b          = min([b m]);

gamma = c_gamma * beta_fi; % step size
tau   = mu * gamma; % prox step-size

% initialise iterate histories
ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);
tk = zeros(floor(maxits/objEvery), 1);

mean_fk     = zeros(floor(maxits/objEvery), 1);
mean_fk_old = 0;

% initialise iterates
x     = x0;
x_old = x0;
l     = 0;

% initialise gradient batches
Gj_k1 = zeros(n,b);
Gj_k2 = zeros(n,b);

% first full gradient
g_old = GradF(x);

its        = 1;
breakcount = 0;

% running print
fprintf(sprintf('performing SARAH...\n'));
fprintf('Running for %d epoch(s) with %d steps per epoch...\n',floor(maxits*q),1/q)

itsprint(sprintf('      step %09d: Objective = %.9e', 1,0), 1);

tic
while(its<=maxits)
    
    if rand(1) < q
            g = GradF(x);
    else
        j = randperm(m,b);

        for batch_num = 1:length(j)
            Gj_k1(:,batch_num) = iGradF(x, j(batch_num));
            Gj_k2(:,batch_num) = iGradF(x_old, j(batch_num));
        end

        g = mean( Gj_k1 - Gj_k2, 2 ) + g_old;
    end
            
    w = x - gamma * g;

    x_old = x;
    g_old = g;

    x = ProxJ(w, tau);

    %%% Compute info
    if mod(its,objEvery)==0
        l = l+1;
        fk(l) = ObjF(x);
        ek(l) = norm(x(:)-x_old(:), 'fro');
        sk(l) = sum(abs(x) > 0);
        gk(l) = gamma;

        mean_fk(l) = mean(fk(max(1,l-window):l));

        if mod(its,printEvery) == 0
            if printObj == 1
                itsprint(sprintf('      step %09d: Mean obj = %.9e\n', its, mean_fk(l)), its); 
            else
                itsprint(sprintf('      step %09d: norm(ek) = %.3e', its, ek(l)), its);
            end
        end

        %%% Stop?
        if abs(mean_fk(l) - mean_fk_old) < tol || abs(mean_fk(l) - mean_fk_old) > 1e10; breakcount=breakcount+1; end
        mean_fk_old = mean_fk(l);

        if breakcount > 10
            fprintf('Breaking due to change in objective value'); break;
        end

    end


     % Save
    if mod(its,saveEvery) == 0
        fprintf('\n Saving... \n')
        save(para.name,'gk','sk','ek','fk','mean_fk','x','tk','para')
        itsprint(sprintf('      step %09d: Objective = %.9e \n', t,fk(l)), 1); 
    end
    
    its = its + 1;
end

fk = fk(1:l);
mean_fk = (1:l);
ek = ek(1:l);
sk = sk(1:l);
gk = gk(1:l);
tk = tk(1:l);

% save(para.name,'gk','sk','ek','fk','mean_fk','tk','x','para')

end





% function to set options
function out = setOpts(options, opt, default)
    if isfield(options, opt)
        out = options.(opt);
    else
        out = default;
    end
end % function: setOpts
