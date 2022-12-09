

x = reshape(Test_X_stack(:,2,:), [1,8]);
y = reshape(Test_Y_stack(:,2,:), [10,8]);


x2 = reshape(Test_X_stack(:,1,:), [1,10]);

Test_X_stack(:,:,9:10) = [];
Test_Y_stack(:,:,9:10) = [];