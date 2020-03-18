function [x, its, ek, fk, sk, tk, gk] = func_SARGE(para, iGradF, ObjF, ProxJ)
%
% Solves the problem
%
%    min_x   1/m sum_{i=1}^m f_i(x) + mu * J(x)
%
% where f_i has a Lipshcitz continuous gradient for all i.
%
% Inputs:
%   para   - struct of parameters
%   iGradF - function handle mapping (x,i) to the gradient of f_i.
%   ObjF   - function handle returning the objective value
%   ProxJ  - function handle returning the proximal operator of the
%            non-smooth term
%
% Outputs:
%   x   - minimiser
%   its - number of iterations
%   ek  - distance between iterations ||x_k - x_{k-1}||_2
%   fk  - objective value
%   sk  - number of non-zero entries of x (size of support)
%   tk  - time
%   gk  - step-size
%
% Parameters:
%     mu         - non-negative tuning parameter ( m^(-1/2) );
%     c_gamma    - step-size times Lipschitz constant of f_i (0.1);
%     beta_fi    - inverse of Lipschitz constant of f_i (1);
%     maxits     - maximum number of iterations (1000);
%     printEvery - number of iterations between printing (100);
%     saveEvery  - number of iterations between saves (100);
%     objEvery   - number of iterations between objective prints (100);
%     printObj   - print objective values? (True);
%     theta      - bias parameter, larger values give more weight
%                  to stored gradients, with a value of 1 corresponding
%                  to the SAGA gradient estimator (1);
%     b          - batch size (1);
%     tol        - tolerance in distance between iterates (1e-4)
%     x0         - starting point (vector with entries drawn i.i.d. from
%                  the standard normal distribution)

% set problem dimensions
if isfield(para,'m') && isfield(para,'n')
    m = para.m;
    n = para.n;
else
    error('Must provide problem dimensions para.m and para.n')
end

% set parameters
mu      = setOpts(para,'mu',1/sqrt(m));
c_gamma = setOpts(para,'c_gamma',0.1);
beta_fi = setOpts(para,'beta_fi',1);
maxits  = setOpts(para,'maxits',1e4);
printEvery = setOpts(para,'printEvery',100);
saveEvery  = setOpts(para,'saveEvery',100);
printObj   = setOpts(para,'printObj',1);
objEvery   = setOpts(para,'objEvery',100);
b          = setOpts(para,'b',1);
tol        = setOpts(para,'tol',1e-4);
x0         = setOpts(para,'x0',zeros(n,1));

% regulate batch size
b          = min([b m]);

% running print
fprintf(sprintf('performing SARGE...\n'));
itsprint(sprintf('      step %09d: Objective = %.9e \n', 1,0), 1); 

gamma = c_gamma * beta_fi; % step size
tau   = mu * gamma; % prox step-size
para  = rmfield(para,'W'); % for better storage in save files

G = zeros(n, m);
for i=1:m
    G(:, i) = 1/m * iGradF(x0, i);
end

mean_grad = mean(G,2);
g_old     = b/m * mean_grad;

% initialise gradient tables
gj_old = zeros(n,b);
gj     = zeros(n,b);

% initialise iterate histories
ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);
tk = zeros(floor(maxits/objEvery), 1);

% initialise x
x     = x0;
x_old = x;
l     = 0;
its   = 1;

tic
while(its<maxits)
    
    j = randperm(m, b);
    
    for batch_num = 1:length(j)
        gj_old(:,batch_num) = G(:, j(batch_num));
        gj(:,batch_num)     = iGradF(x, j(batch_num)) - (1-b/m)*iGradF(x_old, j(batch_num));
        G(:,j(batch_num))   = iGradF(x, j(batch_num)) - (1-b/m)*iGradF(x_old, j(batch_num));
    end
    
    g = mean(gj - gj_old,2) + mean_grad + (1 - b/m) * g_old;

    w = x - gamma * g;
    
    x_old = x;
    g_old = g;
    
    x = ProxJ(w, tau);
    
    mean_grad = mean_grad + 1/m * sum(gj - gj_old,2);
   
    %%% Compute info
    if mod(its,objEvery)==0
        l = l+1;
        fk(l) = ObjF(x);
        ek(l) = norm(x(:)-x_old(:), 'fro');
        sk(l) = sum(abs(x) > 0);
        tk(l) = toc;
        gk(l) = gamma;
        
        if mod(its,printEvery) == 0
            if printObj == 1
                itsprint(sprintf('      step %09d: Objective = %.9e\n', its, fk(l)), its); 
            else
                itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,ek(l)), its);
            end
        end
        
        %%% Stop?
        if ((ek(l))<tol)||(ek(l)>1e10); break; end
        
    end
  
    % Save
    if mod(its,saveEvery) == 0
        fprintf('\n Saving... \n')
        save(para.name,'gk','sk','ek','fk','x','tk','para')
        itsprint(sprintf('      step %09d: Objective = %.9e \n', its,fk(l)), 1); 
    end
    
    its = its + 1;
    
end
fprintf('\n');

if its == maxits
    fprintf('\n Reached maximum number of allowed iterations... \n')
end

fk = fk(1:l);
ek = ek(1:l);
sk = sk(1:l);
tk = tk(1:l);
gk = gk(1:l);



% save(para.name,'gk','sk','ek','fk','x','tk','para')

end





% function to set options
function out = setOpts(options, opt, default)
    if isfield(options, opt)
        out = options.(opt);
    else
        out = default;
    end
end % function: setOpts

