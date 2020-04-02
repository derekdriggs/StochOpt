function [xk1, its, ek, fk, sk, gk] = func_SVRG_LC_noncon(para, GradF, iGradF, ObjF, ProxJ)
%

fprintf('performing SVRG_LC_noncon...\n');
itsprint(sprintf('      step %09d: Objective = %.9e', 1,0), 1);

% parameters
n = para.n;
m = para.m;
gamma  = para.gamma;
tau    = para.tau;

if isfield(para,'eta')
    eta    = para.eta;
end

q      = para.q;
para   = rmfield(para,'gamma');
para   = rmfield(para,'tau');

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





x_tilde = x0;

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%%% Initialise records
ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);

z   = x0;
y   = x0;

l = 0;

mu = GradF(x0);

para.grad_count = 0;

its = 1;
while(its<maxits)
    
    r   = rand(1);
    xk1 = tau(its) * z + (1 - tau(its)) * y;
    
    if r < q || its == 1
        x_tilde = xk1;
        mu = GradF(x_tilde);
        para.grad_count = para.grad_count + 1;
    end

    j = randperm(m, 1);
    
    Gj_k1 = iGradF(xk1, j);
    Gj_k2 = iGradF(x_tilde, j);
    
    % Update y
    if isfield(para,'eta')
        w = xk1 - eta * (Gj_k1 - Gj_k2) - eta * mu;
        y = ProxJ(w, lambda*eta);
    else
        y = tau(its)*z+(1-tau(its))*y;
    end
    
    
    % Update z
    w = z - gamma(its) * (Gj_k1 - Gj_k2) - gamma(its) * mu;
    z = ProxJ(w, lambda*gamma(its));
    
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
