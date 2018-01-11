function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%
                                             % X           300 * 2
% Set K                                      % centroids     3 * 2
K = size(centroids, 1);                      % K  = 3   

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);                   % m*1         300 * 1

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% initial_centroids = [3 3; 6 2; 8 5];

distants = zeros(K,1);                               %  3 * 1
for i = 1:size(X,1)                                  %  1 - m
    val = X(i,:) - centroids;                        %  3 * 2  
    dis_1 = val.^2;                                  %  3 * 2
    for p =1:size(centroids,2)
        distants =distants + dis_1(:,p);             %  3 * 1
    end
    
    distants =sqrt(distants);
    
    for j = 1:length(distants)
        if distants(j) == min(distants)
            idx(i)=j;
        end
    end 
end

% =============================================================

end

