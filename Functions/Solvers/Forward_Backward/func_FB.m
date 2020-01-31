function [x, its, ek, fk, sk, tk, gamma] = func_FB(para, GradF, ObjF, ProxJ)

fprintf(sprintf('performing Forward--Backward...\n'));
itsprint(sprintf('      step %09d: norm(ek) = %.3e', 1,1), 1);

% parameters
n = para.n; 
gamma = para.eta;
tau = para.mu * gamma;

% stop cnd, max iteration
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

% 
%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

%%% obtain the minimizer x^\star
ek = zeros(1, floor(maxits/objEvery));
sk = zeros(1, floor(maxits/objEvery));
fk = zeros(1, floor(maxits/objEvery));
gk = zeros(1, floor(maxits/objEvery));
tk = zeros(1, floor(maxits/objEvery));

x = x0; % xk

l   = 0;
its = 1;

tic

while(its<maxits)
    
    xkm1 = x;
    g = x - gamma*GradF(x);
    x = ProxJ(g, tau);
    
    %%% Compute info
    if mod(its,objEvery)==0
        l = l+1;
        fk(l) = ObjF(x);
        ek(l) = norm(x(:)-xkm1(:), 'fro');
        sk(l) = sum(abs(x) > 0);
        gk(l) = gamma;
        tk(l)  = toc;
        
        if mod(its,printEvery) == 0
            if Obj == 1
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
        save(para.name,'gk','sk','ek','fk','tk','para')
        itsprint(sprintf('      step %09d: Objective = %.9e \n', its,fk(l)), 1); 
    end
    
    its = its + 1;
    
end
fprintf('\n');

fk = fk(1:l);
ek = ek(1:l);
sk = sk(1:l);
gk = gk(1:l);
tk = tk(1:l);

save(para.name,'gk','sk','ek','fk','tk','para')

end