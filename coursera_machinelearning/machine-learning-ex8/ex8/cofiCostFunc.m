function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params             
%     X = n_m*n              % 5*3
% Theta = n_u*n              % 4*3
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...   
                num_users, num_features);             % 4*3

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));                % n_m * n           5*3
Theta_grad = zeros(size(Theta));        % n_u * n           4*3

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features      5*3
%        Theta - num_users  x num_features matrix of user features    4*3
%        Y - num_movies x num_users matrix of user ratings of movies  5*4
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the  5*4
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%  J vector method (without reg)
J = J+ (1/2)*(sum(sum(((X*Theta').*R-Y).^2)));

% for loop method 
% for i =1:num_movies
%     for j = 1:num_users
%         if R(i,j) ==1
%             J = J + (((X(i,:)*Theta(j,:)')-Y(i,j)).^2);
%         end
%     end
% end
% J =J/2 

% J vector method (have reg)
Reg1 = (lambda/2)*sum(sum((X(:,:).^2)));
Reg2 = (lambda/2)*sum(sum(Theta(:,:).^2));
Reg = Reg1 + Reg2;
J = J + Reg;


% Grad without reg

X_grad = X_grad + ((X*Theta').*R-Y)*Theta;              % n_m * n

Theta_grad = Theta_grad + ((X*Theta').*R-Y)'*X;         % n_u * n

% Grad have reg
Grad_X_Reg = (lambda)*X;
Grad_Theta_Reg = (lambda)*Theta;

X_grad = X_grad + Grad_X_Reg;
Theta_grad =Theta_grad + Grad_Theta_Reg ;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
