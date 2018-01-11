function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.

% X          211 * 2
% y          211 * 1
% Xval       200 * 2 
% yval       200 * 1

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

c_s_set = [0.01,0.03,0.1,0.3,1,3,10,30];
% error_set = zeros(8,8);
min_err = nan;
for i = 1:8
    for j = 1:8
        C = c_s_set(i);
        sigma = c_s_set(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model,Xval);
        %error_set(i,j) = mean(double(predictions~=yval));
        err = mean(double(predictions~=yval));
        if (isnan(min_err)||err<min_err)
            min_err =err;
            best_C = C;
            best_S = sigma;
        end
    end
end  
C = best_C;
sigma = best_S;
% for p = 1:8
%     for k = 1:8
%         if error_set(p,k) == min(error_set)
%             best_C = c_s_set(p);
%             best_s = c_s_set(k);
%         end
%     end
% end
% C = best_C;
% sigma = best_s;

% =========================================================================

end






