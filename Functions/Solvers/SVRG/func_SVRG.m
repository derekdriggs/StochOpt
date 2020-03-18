function [x, t, ek, fk, sk, gk] = func_SVRG(para, GradF ,iGradF, ObjF, ProxJ)
%
% Uses SVRG to solve the problem
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
%   sk  - number of non-zero entries of x (size of support)
%   tk  - time
%   gk  - step-size
%
% Parameters:
%     m          - number of functions in smooth component;
%     n          - length of vector x;
%     mu         - non-negative tuning parameter ( n^(-1/2) );
%     P          - epoch length (2*m);
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

% set problem dimensions
if isfield(para,'m') && isfield(para,'n')
    m = para.m;
    n = para.n;
else
    error('Must provide problem dimensions para.m and para.n')
end

% set parameters
mu      = setOpts(para,'mu',1/sqrt(m));
P       = setOpts(para,'P',2*m);
c_gamma = setOpts(para,'c_gamma',0.1);
beta_fi = setOpts(para,'beta_fi',1);
maxits  = setOpts(para,'maxits',1e4);
printEvery = setOpts(para,'printEvery',100);
saveEvery  = setOpts(para,'saveEvery',100);
printObj   = setOpts(para,'printObj',1);
objEvery  = setOpts(para,'objEvery',100);
theta      = setOpts(para,'theta',1);
b          = setOpts(para,'b',1);
tol        = setOpts(para,'tol',1e-4);
x0         = setOpts(para,'x0',randn(n,1));

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

% initialise iterates
x       = x0;
x_tilde = x;
l       = 0;

% initialise gradient batches
Gj_k1 = zeros(n,b);
Gj_k2 = zeros(n,b);

its  = 1;
t    = 1;
Conv = 0; % break if converged (Conv = 1)

% running print
fprintf(sprintf('performing SVRG...\n'));
fprintf('Running for %d epoch(s) with %d steps per epoch...\n',floor(maxits/P),P)

itsprint(sprintf('      step %09d: Objective = %.9e', 1,0), 1);

tic
while(its<=floor(maxits/P))
    
    % fprintf('computing new mu')
    mu = GradF(x_tilde);
    
    for p=1:P
        
        x_old = x;
        
        j = randperm(m,b);
        
        for batch_num = 1:length(j)
            Gj_k1(:,batch_num) = iGradF(x, j(batch_num));
            Gj_k2(:,batch_num) = iGradF(x_tilde, j(batch_num));
        end

        w = x - gamma / theta * mean(Gj_k1 - Gj_k2, 2) - gamma * mu;
        x = ProxJ(w, tau);
        
        %%% Compute info
        if mod(t,objEvery)==0
            l = l+1;
            fk(l) = ObjF(x);
            ek(l) = norm(x(:)-x_old(:), 'fro');
            sk(l) = sum(abs(x) > 0);
            gk(l) = gamma;
            tk(l) = toc;
            
            if mod(t,printEvery) == 0
                if printObj == 1
                    itsprint(sprintf('      step %09d: Objective = %.9e\n', t, fk(l)), t); 
                else
                    itsprint(sprintf('      step %09d: norm(ek) = %.3e', t,ek(l)), t);
                end
            end

            %%% Stop?
            if ((ek(l))<tol)||(ek(l)>1e10); fprintf('Breaking due to change in iterate value \n'); Conv = 1; break; end

        end
        
        
        % Save
        if mod(t,saveEvery) == 0
            fprintf('\n Saving... \n')
            save(para.name,'gk','sk','ek','fk','x','tk','para')
            itsprint(sprintf('      step %09d: Objective = %.9e \n', t,fk(l)), 1); 
        end
        
        t = t+1;
        
    end
    
    if Conv
            break
    end
        
    x_tilde = x;
    
    
    its = its + 1;
        
end

if its >= floor(maxits/P)
    fprintf('\n Reached maximum number of allowed iterations... \n')
end

fk = fk(1:l);
ek = ek(1:l);
sk = sk(1:l);
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
