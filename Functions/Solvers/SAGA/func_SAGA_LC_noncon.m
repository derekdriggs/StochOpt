function [xk1, its, ek, fk, sk, gk] = func_SAGA_LC_noncon(para, iGradFOpt, ObjF, ProxJ)
%

fprintf('performing SAGALC_noncon...\n');
itsprint(sprintf('      step %09d: Objective = %.9e', 1,0), 1);

% parameters
n = para.n;
m = para.m;
gamma  = para.gamma;
tau    = para.tau;
para   = rmfield(para,'gamma');
para   = rmfield(para,'tau');

W      = para.W;

para = rmfield(para,'W');

lambda = para.mu;

% stop cnd, max iteration
tol    = para.tol;
maxits = para.maxits;
Obj    = para.Obj;

% initial point
if isfield(para,'x0')
    x0 = para.x0;
else
    x0 = zeros(n, 1);
end

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


G = zeros(1, m);
for i=1:m
    G(:, i) = iGradFOpt(x0, i);
end

mean_grad = 1/m * W' * G';

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%%% obtain the minimizer x^\star
ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);

z = x0;
y = x0;

l = 0;

its = 1;
while(its<maxits)
    
    xk1 = tau(its) * z + (1 - tau(its)) * y;
    
    % j = mod(its-1, m) + 1;
    j = randperm(m, 1);
    
    gj_old    = W(j,:)' * G(:, j);
    gj = iGradFOpt(xk1, j);
    
    G(:,j) = gj;
    
    gj = W(j,:)' * gj;
    
    % Update y
    %w = xk1 - eta * (gj - gj_old) - eta * mean_grad;
    %y = ProxJ(w, lambda*eta);
    
    
    % Update z
    w = z - gamma(its) * (gj - gj_old) - gamma(its) * mean_grad;
    z = ProxJ(w, lambda*gamma(its));
    
    % Update y
    y     = tau(its)*z + (1-tau(its))*y;
    z_old = z;
    
    
    % Update average of stored gradients
    mean_grad = mean_grad - 1/m*gj_old + 1/m*gj;
    
    %%% Compute info
    if mod(its,objEvery)==0
        l = l+1;
        fk(l) = ObjF(xk1);
        ek(l) = norm(xk1(:)-(tau(its)*z+(1-tau(its))*y), 'fro');
        sk(l) = sum(abs(xk1) > 0);
        gk(l) = gamma(its);
        
        if Obj == 1
            itsprint(sprintf('      step %09d: Objective = %.9e\n', its, fk(l)), its); 
        else
            itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,ek(l)), its);
        end
        
        %%% Stop?
        if ((ek(l))<tol)||(ek(l)>1e10); break; end
    end
 
    
    % Save
    if mod(its,saveEvery) == 0
        fprintf('\n Saving... \n')
        save(para.name,'gk','sk','ek','fk','xk1','para')
        itsprint(sprintf('      step %09d: Objective = %.9e \n', its,fk(l)), 1); 
    end
    
    its = its + 1;
    
end
fprintf('\n');

l = l+1;
fk(l) = ObjF(xk1);
ek(l) = norm(xk1(:)-(tau(its)*z+(1-tau(its))*y), 'fro');
sk(l) = sum(abs(xk1) > 0);
gk(l) = gamma(its);

fk = fk(1:l);
ek = ek(1:l);
sk = sk(1:l);
gk = gk(1:l);

% save(para.name,'gk','sk','ek','fk','xk1','para')

end