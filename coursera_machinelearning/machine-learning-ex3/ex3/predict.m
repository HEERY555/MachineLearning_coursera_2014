function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);                   % num of X              5000
num_labels = size(Theta2, 1);     % num_labels = num of Theta2's row  10 

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);         %  m*1     5000*1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%      n = num of classify
% 

% Theta1    (num of Theta2's units) * m     25*(400+1)
% Theta2    (n+1)*(num of Theta2's units+1) 10*(25+1)
X = [ones(m,1),X];                   % m * (n+1)    5000 * 401
%layer_2_units = size(Theta1,1);     % num of Theta's units  25
%z = zeros(layer_2_units) ;          % 25* 401  

a = sigmoid(X*Theta1');              % 5000 * 25
a = [ones(size(a,1),1),a];           % 5000 * 26
pre_sig = (sigmoid(a*Theta2'))  ;    % 5000*10
for i = 1:m
    [~,p(i)] = max(pre_sig(i,:));

% =========================================================================


end
