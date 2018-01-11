function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples  12

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));       % 2 *1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%                      theta      2 * 1
%                        X       21 * 2


%  Compute cost
pre = X * theta;                 % 21 * 1
sqr_error = (pre - y).^2;        % 21 * 1

J =  J + (1/(2*m))*(sum(sqr_error));  %  Cost Function (No Reg)

Reg = (lambda/(2*m))*(sum(theta.^2)-(theta(1).^2));
J = J + Reg;                          %   Cost Function  (Add Reg)



% compute the grad 
theta_1 = theta;
theta_1(1) = 0;
grad = grad + (1/m)*(  (X'*(pre-y)) + lambda* theta_1);

% =========================================================================

grad = grad(:);

end
