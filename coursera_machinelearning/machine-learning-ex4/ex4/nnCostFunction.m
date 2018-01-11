function [J grad] = nnCostFunction(nn_params, ...   
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network  
                         % (1:25*(400+1)), 25,(400+1)               25 * 401
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
                           % ((1+(25*(400+1))):end)   10,(25+1)       10*26
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);                                % 5000
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));              %  25 * 401
Theta2_grad = zeros(size(Theta2));              %  10 * 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% Theta1      25 * 401
% Theta2      10 * 26 
% nn_params [Theta1(:) ; Theta2(:)]   
% input_layer_size   400
% hidden_layer_size  25 
% num_labels         10
% X    5000 * 400
% y    5000 * 1

K = num_labels;                    % 10


% made the Y matrix    5000 * 10    every y is the vector     1 * 10 
Y = zeros(m,K);                     % 5000*10

for i =1:m 
    Y(i,y(i)) = 1;                 %  i represent the columns and y(i) =1 represent the i'th value is 1               
end

A1 = [ones(m,1),X];                % 5000 * (400+1)
a2 = sigmoid(A1 *Theta1');         % 5000 * 25
A2 = [ones(m,1),a2];               % 5000 * 26
A3 = sigmoid(A2 *Theta2');         % 5000 * 10

cost = -Y(:)' * log(A3(:)) - (1-Y(:))' * log(1-A3(:));
% Feedforward and Cost Function
J =  J + (1/m) * cost;

% Regularized Cost Function
%  (lambda /(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); 
Reg = (lambda/(2*m)) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + Reg;         


% backpropagation
delta_3 = A3 - Y ;                                                   % 5000 * 10
delta_2 = (delta_3*Theta2(:,2:end)).*sigmoidGradient(A1 *Theta1');   % 5000 * 25

Theta1_grad = Theta1_grad + (1/m)* ((delta_2'*A1));                  % 25 * 401 
Theta2_grad = Theta2_grad + (1/m)* ((delta_3'*A2));                  % 10 * 26

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));              
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end