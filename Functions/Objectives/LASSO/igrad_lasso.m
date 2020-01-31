function grad = igrad_lasso(x, i, W, Y)

    grad = W(i,:)'*(W(i,:)*x - Y(i));

end