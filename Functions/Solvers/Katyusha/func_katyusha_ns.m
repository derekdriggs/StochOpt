function [xk1, its, ek, fk, sk, tk, gk] = func_katyusha_ns(para, GradF, iGradF, ObjF, ProxJ)
%
% Uses Katyusha_ns to solve the problem
%
%    min_x   1/n sum_{i=1}^n f_i(x) + mu * J(x)
%
% where f_i has a Lipschitz continuous gradient for all i and J is
% non-strongly convex.
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
%     gamma      - step-size for mirror descent step, a function handle (0.1);
%     eta        - step-size for gradient descent step (0.1);
%     tau1       - (Nesterov) momentum parameter, a function handle (0.25);
%     tau2       - (Katyusha) momentum parameter, a constant (0.5);
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
gamma   = setOpts(para,'gamma',@(x)0.1);
eta     = setOpts(para,'eta',0.1);
tau1    = setOpts(para,'tau1',@(x)0.25);
tau2    = setOpts(para,'tau2',0.5);
maxits  = setOpts(para,'maxits',1e4);
printEvery = setOpts(para,'printEvery',100);
saveEvery  = setOpts(para,'saveEvery',100);
printObj   = setOpts(para,'printObj',1);
objEvery   = setOpts(para,'objEvery',100);
b          = setOpts(para,'b',1);
tol        = setOpts(para,'tol',1e-4);
x0         = setOpts(para,'x0',randn(n,1));

% regulate batch size
b          = min([b m]);

try
    tau1(1)
catch
    error('tau must be a function handle')
end

try
    gamma(1)
catch
    error('gamma must be a function handle')
end

% regulate momentum parameters
if tau1(1) + tau2 > 1
    error('tau1 + tau2 is too large')
end

% initialise iterate histories
ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);
tk = zeros(floor(maxits/objEvery), 1);

% initialise iterates
x       = x0;
z       = x0;
y       = x0;
x_tilde = x0;
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
while(its<maxits)
    
    g_full = GradF(x_tilde);
    
    for p=1:P
        
        xk1 = tau1(its)*z + tau2*x_tilde + (1-tau1(its)-tau2)*y;
        
        j = randperm(m,b);
        
        for batch_num = 1:length(j)
            Gj_k1(:,batch_num) = iGradF(x, j(batch_num));
            Gj_k2(:,batch_num) = iGradF(x_tilde, j(batch_num));
        end
        
        
        % Update y
        w = xk1 - eta * mean( Gj_k1 - Gj_k2, 2 ) - eta * g_full;
        y = ProxJ(w, mu*eta);
        
        
        % Update z
        w = z - gamma(its) * mean( Gj_k1 - Gj_k2, 2 ) - gamma(its) * g_full;
        z = ProxJ(w, gamma(its)*mu);
       
        
        %%% Compute info
        if mod(t,objEvery)==0
            l = l+1;
            fk(l) = ObjF(xk1);
            ek(l) = norm(xk1(:)-(tau1(its)*z+tau2*x_tilde+(1-tau1(its)-tau2)*y), 'fro');
            sk(l) = sum(abs(xk1) > 0);
            gk(l) = gamma(its);
            tk(l) = toc;

            if mod(t,printEvery) == 0
                if printObj == 1
                    itsprint(sprintf('      step %09d: Objective = %.9e\n', t, fk(l)), t); 
                else
                    itsprint(sprintf('      step %09d: norm(ek) = %.3e', t,ek(l)), t);
                end
            end

            %%% Stop?
            if ((ek(l))<tol)||(ek(l)>1e10); fprintf('Breaking due to change in iterate value'); Conv = 1; break; end

        end
        
        
        % Save
        if mod(t,saveEvery) == 0
            fprintf('\n Saving... \n')
            save(para.name,'gk','sk','ek','fk','xk1','tk','para')
            itsprint(sprintf('      step %09d: Objective = %.9e \n', t,fk(l)), 1); 
        end
        
        t = t + 1;
        
    end
    
    x_tilde = xk1; % can modify x_tilde update to average, weighted average
    
    its = its + 1;
    
    if Conv; break; end
    
end    
fprintf('\n');

if its == maxits
    fprintf('\n Reached maximum number of allowed iterations... \n')
end

ek = ek(1:l);
sk = sk(1:l);
gk = gk(1:l);
fk = fk(1:l);
tk = tk(1:l);

% save(para.name,'gk','sk','ek','fk','xk1','tk','para')

end