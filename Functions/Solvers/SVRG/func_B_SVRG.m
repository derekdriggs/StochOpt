function [x, t, ek, fk, sk, gk] = func_B_SVRG(para, GradF ,iGradF, ObjF, ProxJ)
%

fprintf(sprintf('performing SVRG...\n'));

% parameters
P = para.P;
m = para.m;
n = para.n;
gamma = para.c_gamma * para.beta_fi;
tau   = para.mu * gamma;
theta = para.theta; % bias parameter

% stop cnd, max iteration
tol = para.tol;
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

ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);

x = x0;
x_tilde = x;

l = 0;

its = 1;
t = 1;

Conv = 0;

fprintf('Running for %d epoch(s) with %d steps per epoch...\n',floor(maxits/P),P)

itsprint(sprintf('      step %09d: Objective = %.9e', 1,0), 1);

while(its<=floor(maxits/P))
    
    % fprintf('computing new mu')
    mu = GradF(x_tilde);
    
    
    for p=1:P
        
        x_old = x;
        
        j = randperm(m,1);
        
        Gj_k1 = iGradF(x, j);
        Gj_k2 = iGradF(x_tilde, j);

        w = x - (gamma / theta) * ( Gj_k1 - Gj_k2 ) - gamma * mu;
        x = ProxJ(w, tau);
        
        %%% Compute info
        if mod(t,objEvery)==0
            l = l+1;
            fk(l) = ObjF(x);
            ek(l) = norm(x(:)-x_old(:), 'fro');
            sk(l) = sum(abs(x) > 0);
            gk(l) = gamma;

            if mod(t,printEvery) == 0
                if Obj == 1
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
            save(para.name,'gk','sk','ek','fk','x','para')
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

fk = fk(1:l);
ek = ek(1:l);
sk = sk(1:l);
gk = gk(1:l);

% save(para.name,'gk','sk','ek','fk','x','para')

end
