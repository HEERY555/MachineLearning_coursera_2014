function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%
%  X 12 * 2
%  y 12 * 1
%  Xval 21*2
%  yval 21*1
%  lambda = 0

% Number of training examples
m = size(X, 1);                   % m = 12 

% You need to return these values correctly
error_train = zeros(m, 1);       % 12 * 1
error_val   = zeros(m, 1);       % 12 * 1

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
% for i = 1 : m
%     theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
 %   error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
 %   error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
%end



% When you are computing the training set error,make sure you compute it 
% on the training subset (i.e., X(1:n,:) and y(1:n))¡¾instead of the entire training set¡¿

% However, for the cross validation error,
% you should compute it over the ¡¾entire cross validation set¡¿


for i = 1 : m
     theta = trainLinearReg(X(1:i,:), y(1:i), lambda);
     train_pre = X * theta;                                   % 12 * 1 
     val_pre = Xval * theta;                                  % 12 * 1
     sqr_err_train = (train_pre(1:i)-y(1:i)).^2;              % 12 * 1
     sqr_err_val = (val_pre-yval).^2;                         % 12 * 1 
     error_train(i) = (1/(2*i)) * (sum(sqr_err_train));       %¡¾one part¡¿
     error_val(i) = (1/(2*m))* (sum(sqr_err_val));            %¡¾entire cross validation set¡¿
end

% -------------------------------------------------------------

% =========================================================================

end
