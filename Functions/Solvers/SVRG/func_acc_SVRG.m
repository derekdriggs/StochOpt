function [x, t, ek, fk, sk, gk, vk] = func_acc_SVRG(para, GradF,iGradF, ObjF)
%

fprintf(sprintf('performing acc-SVRG...\n'));
itsprint(sprintf('      step %09d: norm(ek) = %.3e', 1,1), 1);


% parameters
P = para.P;
m = para.m;
n = para.n;
gamma0 = para.c_gamma * para.beta_fi;
gamma = gamma0;
tau = para.mu * gamma;

W = para.W;

% stop cnd, max iteration
tol = para.tol;
maxits = para.maxits;

% initial point
x0 = zeros(n, 1);

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

ek = zeros(maxits, 1);
sk = zeros(maxits, 1);
gk = zeros(maxits, 1);
fk = zeros(maxits, 1);
vk = zeros(maxits, 1);

g_flag = 0;

x = x0;
x_tilde = x;

l = 0;

its = 1;
t = 1;
while(its<maxits/m)
    
    mu = GradF(x_tilde);
    
    x = x_tilde;
    for p=1:P
        
        x_old = x;
        
        j = randsample(1:m,1);
        
        Gj_k1 = iGradF(x, j);
        Gj_k2 = iGradF(x_tilde, j);
        
        w = x - gamma* ( Gj_k1 - Gj_k2 + mu );
        x = wthresh(w, 's', tau);
        
        x(end) = w(end);
        
        distE = norm(x(:)-x_old(:), 'fro');
        if mod(t,1e3)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e', t,distE), t); end
        
        if mod(t, m)==0
            l = l + 1;
            fk(l) = ObjF(x);
        end
        
        sk(t) = sum(abs(x) > 0);
        gk(t) = gamma;
        ek(t) = distE;
                
        if (t>m)&&(mod(t, m)==0)&&(sk(t)<n/1.5)&&(var(sk(t-m+1:t-1))<1)
            PT = diag(double(abs(x)>0));
            WT = W*PT;
            b = zeros(m, 1);
            for i=1:m
                WTi = WT(i,:);
                b(i) = norm(WTi)^2 /4;
            end
            beta_fi_new = 1 /max(b);
            
            E = beta_fi_new /para.beta_fi;
            g_flag = 1;
            
        end

        if (t>50*m)
            vk(t) = var(sk(t-m+1:t));
        end
        
        if (g_flag)&&(mod(t, m)==0)
            
            gamma = min(gamma*1.5, E*gamma0);
            tau = para.mu * gamma;
            
        end

        t = t + 1;
        
    end
    
    x_tilde = x;
    
    %%% stop?
    normE = norm(x_tilde(:)-x_old(:), 'fro');
    if ((normE)<tol)||(normE>1e10); break; end
    
    its = its + 1;
    
end
fprintf('\n');

% g_flag

sk = sk(1:t-1);
ek = ek(1:t-1);
gk = gk(1:t-1);
vk = vk(1:t-1);
fk = fk(1:l);