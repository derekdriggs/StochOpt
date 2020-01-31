function [x, its, dk, sk, gamma] = func_FB_dk(para, GradF, ProxJ, xsol)

fprintf(sprintf('performing Forward--Backward...\n'));
itsprint(sprintf('      step %09d: norm(ek) = %.3e', 1,1), 1);

% parameters
n = para.n; 
gamma = para.c_gamma * para.beta;
tau = para.mu * gamma;

% stop cnd, max iteration
ToL = 1e-12;
maxits = 1e6;

% Forward-Backward Step
FB = @(g, tau) ProxJ(g, tau);

% inertial point
x0 = zeros(n, 1);

% 
%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

%%% obtain the minimizer x^\star
dk = zeros(1, maxits);
sk = zeros(1, maxits);

x = x0; % xk

its = 1;
while(its<maxits)
    
    xkm1 = x;
    
    g = x - gamma*GradF(x);
    x = FB(g, tau);
    x(end) = g(end);
    
    %%% stop?
    normE = norm(x(:)-xsol(:), 'fro');
    if mod(its,1e2)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,normE), its); end
    
    dk(its) = normE;
    if ((normE)<ToL)||(normE>1e10); break; end
        
    sk(its) = sum(abs(x) > 0);
    
    its = its + 1;
    
end
fprintf('\n');

dk = dk(1:its-1);
sk = sk(1:its-1);