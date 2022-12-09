

% i = 1;
% close all;

for i = 1:29
    figure; plot(nonzeros(X_stack_original(:,i)), nonzeros(Y_stack_original(:,i))); axis([50 200 -120 0]);
end

% time(
