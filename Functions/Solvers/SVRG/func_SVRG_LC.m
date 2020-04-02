function [xk1, its, ek, fk, sk, gk] = func_SVRG_LC(para, GradF, iGradF, ObjF, ProxJ)
%

fprintf('performing SVRG_LC...\n');
itsprint(sprintf('      step %09d: Objective = %.9e', 1,0), 1);

% parameters
n = para.n;
m = para.m;
gamma  = para.gamma;
tau    = para.tau;
%eta    = para.eta;
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





%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%%% Initialise records
ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);

z  = x0;
y  = x0;
x_tilde = x0;

mu = GradF(x_tilde);

l = 0;

para.grad_count = 0;

its = 1;
while(its<maxits)
    
    r   = rand(1);
    xk1 = tau * z + (1 - tau) * y;
    
    if r < q
        x_tilde = xk1;
        mu = GradF(x_tilde);
        para.grad_count = para.grad_count + 1;
    end

    j = randperm(m, 1);
    
    Gj_k1 = iGradF(xk1, j);
    Gj_k2 = iGradF(x_tilde, j);
    
    % Update y
    %w = xk1 - eta * (Gj_k1 - Gj_k2) - eta * mu;
    %y = ProxJ(w, lambda*eta);
    
    % Update z
    w = z - gamma * (Gj_k1 - Gj_k2) - gamma * mu;
    z = ProxJ(w, lambda*gamma);
    
    % Update y
    y = tau*z + (1-tau)*y;
    
     %%% Compute info
    if mod(its,objEvery)==0
        l = l+1;
        fk(l) = ObjF(xk1);
        ek(l) = norm(xk1(:)-(tau*z+(1-tau)*y), 'fro');
        sk(l) = sum(abs(xk1) > 0);
        gk(l) = gamma;
        
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
ek(l) = norm(xk1(:)-(tau*z+(1-tau)*y), 'fro');
sk(l) = sum(abs(xk1) > 0);
gk(l) = gamma;

% save(para.name,'gk','sk','ek','fk','xk1','para')

end