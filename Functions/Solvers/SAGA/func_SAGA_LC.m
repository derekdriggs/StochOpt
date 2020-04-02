function [xk1, its, ek, fk, sk, tk, gk] = func_SAGA_LC(para, iGradFOpt, ObjF, ProxJ)
%

fprintf('performing SAGALC...\n');
itsprint(sprintf('      step %09d: Objective = %.12e', 1,0), 1);

% parameters
n = para.n;
m = para.m;
gamma  = para.gamma;
tau    = para.tau;
b      = para.b;   % batch size

if isfield(para,'eta')
    eta    = para.eta;
end

lambda = para.mu;
W      = para.W;

para = rmfield(para,'gamma');
para = rmfield(para,'tau');


% stop cnd, max iteration, print objective?
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

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%%% obtain the minimizer x^\star
ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);
tk = zeros(floor(maxits/objEvery), 1);

z = x0;
y = x0;

l = 0;

its = 1;

tic

while(its<maxits)
    
    xk1 = tau * z + (1 - tau) * y;
    
    % j = mod(its-1, m) + 1;
    j = randperm(m, b);
    
    gj_old = W(j,:)' * G(j)';
    gj     = iGradFOpt(xk1, j);
    G(j) = gj;
    
    gj = W(j,:)' * gj;
    
    
    % Update z
    w = z - (gamma / b) * (gj - gj_old) - gamma * mean_grad;
    z = ProxJ(w, lambda*gamma);
    
    % Update y
    if isfield(para,'eta')
        w = xk1 - eta * (gj - gj_old) - eta * mean_grad;
        y = ProxJ(w, lambda*eta);
    else
        y = tau*z + (1-tau)*y;
    end
    
    mean_grad = mean_grad - 1/m*gj_old + 1/m*gj;
    
    %%% Compute info
    if mod(its,objEvery)==0
        l = l+1;
        fk(l) = ObjF(z);
        ek(l) = norm(xk1(:)-(tau*z+(1-tau)*y), 'fro');
        sk(l) = sum(abs(xk1) > 0);
        gk(l) = gamma;
        tk(l) = toc;
        
        if Obj == 1
            itsprint(sprintf('      step %09d: Objective = %.12e\n', its, fk(l)), its); 
        else
            itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,ek(l)), its);
        end
        
        %%% Stop?
        if ((ek(l))<tol)||(ek(l)>1e10); break; end
    end
 
    
    % Save
    if mod(its,saveEvery) == 0
        fprintf('\n Saving... \n')
        save(para.name,'gk','sk','ek','fk','xk1','tk','para')
        itsprint(sprintf('      step %09d: Objective = %.12e \n', its,fk(l)), 1); 
    end
    
    its = its + 1;
    
end
fprintf('\n');

fk = fk(1:l);
ek = ek(1:l);
sk = sk(1:l);
gk = gk(1:l);
tk = tk(1:l);



% save(para.name,'gk','sk','ek','fk','xk1','tk','para')

end