function [x, its, ek, fk, mean_fk, sk, gk] = func_B_SAGA_noncon_Lin(para, iGradFOpt, ObjF, ProxJ)
%

fprintf(sprintf('performing SAGA for nonconvex objectives...\n \n'));
itsprint(sprintf('      step %09d: Objective = %.9e \n', 1,0), 1); 

% parameters
n = para.n;
m = para.m;
gamma = para.c_gamma * para.beta_fi;
tau = para.mu * gamma;
W   = para.W;
theta = para.theta; % bias parameter

% record mean objective over m iterations
window = m; 

para = rmfield(para,'W');

% stop cnd, max iteration, print objective
tol    = para.tol;
maxits = para.maxits;
Obj    = para.Obj;

% How often to compute objective
if isfield(para,'objEvery')
    objEvery = para.objEvery;
else
    objEvery = floor(maxits*1e-6);
end

% How often to save
if isfield(para,'saveEvery')
    saveEvery = para.saveEvery;
else
    saveEvery = 1e9;
end



% How often to print
if isfield(para,'printEvery')
    printEvery = para.printEvery;
else
    printEvery = objEvery;
end





% initial point
if isfield(para,'x0')
    x0 = para.x0;
else
    x0 = zeros(n, 1);
end

G = zeros(1, m);
for i=1:m
    G(:, i) = iGradFOpt(x0, i);
end

mean_grad = 1/m * W' * G';

mean_fk_old  = 0;

%%% obtain the minimizer x^\star
ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);

mean_fk = zeros(floor(maxits/objEvery), 1);

x = x0; % xk

l = 0;

its = 1;
while(its<maxits)
    
    x_old = x;
    
    % j = mod(its-1, m) + 1;
    j = randperm(m, 1);
    
    gj_old = W(j,:)' * G(:, j);
    gj     = iGradFOpt(x_old, j);
    G(:,j) = gj;
    
    gj = W(j,:)' * gj;
    
    w = x - (gamma / theta) * (gj - gj_old) - gamma * mean_grad;
    
    x = ProxJ(w, tau);
    
    mean_grad = mean_grad - 1/m*gj_old + 1/m*gj;
    
    %%% Compute info
    if mod(its,objEvery)==0
        l = l+1;
        fk(l) = ObjF(x);
        ek(l) = norm(x(:)-x_old(:), 'fro');
        sk(l) = sum(abs(x) > 0);
        gk(l) = gamma;
        
        mean_fk(l) = mean(fk(max(1,l-window):l));
        
        if mod(its,printEvery) == 0
            if Obj == 1
                itsprint(sprintf('      step %09d: Mean objective = %.9e\n', its, mean_fk(l)), its); 
            else
                itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,ek(l)), its);
            end
        end
        
        %%% Stop?
        if abs(mean_fk(l) - mean_fk_old) < tol || abs(mean_fk(l) - mean_fk_old) > 1e10; break; end
        
        mean_fk_old = mean_fk(l);
    end
 
    
    % Save
    if mod(its,saveEvery) == 0
        fprintf('\n Saving... \n')
        save(para.name,'gk','sk','ek','fk','mean_fk','x','para')
        itsprint(sprintf('      step %09d: Objective = %.9e \n', its,fk(l)), 1); 
    end
    
    its = its + 1;
    
end
fprintf('\n');

mean_fk = mean_fk(1:l);

fk = fk(1:l);
ek = ek(1:l);
sk = sk(1:l);
gk = gk(1:l);



save(para.name,'gk','sk','ek','fk','mean_fk','x','para')

end
