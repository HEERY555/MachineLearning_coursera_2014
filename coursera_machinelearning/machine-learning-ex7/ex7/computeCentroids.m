function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);                    % X 300*2   m=300  n=2 

% You need to return the following variables correctly.
centroids = zeros(K, n);            % 3*2


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
%cluster_1_count = zeros(1,n)

count_sum = zeros(K,n);                     % 3 * 2 
count = zeros(K,1);                         % count num of all classify vector  3*1
for i = 1:length(idx)                       % i  1-300
    for j =1:K                              % j  1-3 
        if idx(i)==j
            count(j)=count(j)+1;
            count_sum(j,:) = count_sum(j,:)+X(i,:);
            centroids(j,:) = count_sum(j,:)/count(j);
        end
    end 
end






% =============================================================


end

