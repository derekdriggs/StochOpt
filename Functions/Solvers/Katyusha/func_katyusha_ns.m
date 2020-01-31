function [xk1, its, ek, fk, sk, gk] = func_katyusha_ns(para, GradF, iGradF, ObjF, ProxJ)
%

fprintf('performing Katyusha...\n');

% parameters
P = para.P;
n = para.n;
m = para.m;
gamma  = para.gamma;
tau1   = para.tau1;
tau2   = para.tau2;
if tau1(1) + tau2 > 1
    error('tau1 is too large')
end
% eta    = para.eta;
eta    = para.eta;
lambda = para.mu;

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
x0 = zeros(n, 1);

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

ek = zeros(floor(maxits/objEvery), 1);
sk = zeros(floor(maxits/objEvery), 1);
gk = zeros(floor(maxits/objEvery), 1);
fk = zeros(floor(maxits/objEvery), 1);

z = x0;
y = x0;

x_tilde = x0;
x_old = x0;
l = 0;

its = 1;
t = 1;

Broken = 0;

fprintf('Running for %d epoch(s) with %d steps per epoch...\n',floor(maxits/P),P)

itsprint(sprintf('      step %09d: Objective = %.9e', 1,0), 1);

while(its<maxits)
    
    mu = GradF(x_tilde);
    
    for p=1:P
        
        xk1 = tau1(its)*z + tau2*x_tilde + (1-tau1(its)-tau2)*y;
        
        j = randperm(m,1);
        
        Gj_k1 = iGradF(xk1, j);
        Gj_k2 = iGradF(x_tilde, j);
        
        
        % Update y
        w = xk1 - eta* ( Gj_k1 - Gj_k2 + mu );
        y = ProxJ(w, lambda*eta);
        
        
        % Update z
        w = z - gamma(its)* ( Gj_k1 - Gj_k2 + mu );
        z = ProxJ(w, gamma(its)*lambda);
       
        
        %%% Compute info
        if mod(t,objEvery)==0
            l = l+1;
            fk(l) = ObjF(xk1);
            ek(l) = norm(xk1(:)-(tau1(its)*z+tau2*x_tilde+(1-tau1(its)-tau2)*y), 'fro');
            sk(l) = sum(abs(xk1) > 0);
            gk(l) = gamma(its);

            if mod(t,printEvery) == 0
                if Obj == 1
                    itsprint(sprintf('      step %09d: Objective = %.9e\n', t, fk(l)), t); 
                else
                    itsprint(sprintf('      step %09d: norm(ek) = %.3e', t,ek(l)), t);
                end
            end

            %%% Stop?
            if ((ek(l))<tol)||(ek(l)>1e10); fprintf('Breaking due to change in iterate value'); Broken = 1; break; end

        end
        
        
        % Save
        if mod(t,saveEvery) == 0
            fprintf('\n Saving... \n')
            save(para.name,'gk','sk','ek','fk','xk1','para')
            itsprint(sprintf('      step %09d: Objective = %.9e \n', t,fk(l)), 1); 
        end
        
        
        % Keep track of old iterates for next x_tilde update
        x_old = x_old + 1/P * y;
        
        t = t + 1;
        
    end
    x_tilde = x_old;
    x_old   = zeros(size(x_old));
    % x_tilde = xk1; %x_old;
    % x_old   = 1/P * y;
    
    its = its + 1;
    
    if Broken
        break
    end
    
    
    
end    
fprintf('\n');

ek = ek(1:l);
sk = sk(1:l);
gk = gk(1:l);
fk = fk(1:l);

% save(para.name,'gk','sk','ek','fk','xk1','para')

end