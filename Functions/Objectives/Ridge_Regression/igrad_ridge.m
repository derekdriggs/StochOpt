function grad = igrad_ridge(x, i, W, Y, lambda)

    m = size(W,1);
    
    if i <= m
        grad = W(i,:)'*(W(i,:)*x - Y(i));
    else
        grad = lambda * x(i);
    end

end