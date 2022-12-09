function [C,X,Y] = cheby(f,n,p,time)
%---------------------------------------------------------------------------
%CHEBY   Construction of the collocation polynomial.
%        The method is Chebyshev interpolation.
% Sample calls
%   [C] = cheby('f',n)
%   [C,X,Y] = cheby('f',n)
%   [C,X,Y] = cheby('f',n,a,b)
% Inputs
%   f   name of the function 
%   n   degree of the Chebyshev interpolation polynomial
%   a   left  endpoint of the interval
%   b   right endpoint of the interval
% Return
%   C   coefficient list for the Chebyshev interpolation polynomial
%   X   abscissas for interpolation nodes
%   Y   ordinates for interpolation nodes
%
% NUMERICAL METHODS: MATLAB Programs, (c) John H. Mathews 1995
% To accompany the text:
% NUMERICAL METHODS for Mathematics, Science and Engineering, 2nd Ed, 1992
% Prentice Hall, Englewood Cliffs, New Jersey, 07632, U.S.A.
% Prentice Hall, Inc.; USA, Canada, Mexico ISBN 0-13-624990-6
% Prentice Hall, International Editions:   ISBN 0-13-625047-5
% This free software is compliments of the author.
% E-mail address:      in%"mathews@fullerton.edu"
%
% Algorithm 4.6 (Chebyshev Approximation).
% Section	4.5, Chebyshev Polynomials, Page 246
%---------------------------------------------------------------------------
% if nargin==2

%     a = -1.0001;
%     b = 1.0001; 
a = min(time);
b = max(time);
% end

d = pi/(2*n+2);
C = zeros(1,n+1);
a1 = p(1);
a2 = p(2);
a3 = p(3);
a4 = p(4);
a5 = p(5);
for k=1:n+1
  X(k) = cos((2*k-1)*d);   % 0.9511    0.5878    0.0000   -0.5878   -0.9511, 
end
X = (b-a)*X/2+(a+b)/2;  % 1.1603    0.7171    0.0000   -0.7171   -1.1603   /   2.0801
x = X;
Y = (eval(f));

for k = 1:n+1
  z = (2*k-1)*d;
  for j = 1:n+1
    C(j) = C(j) + Y(k)*cos((j-1)*z);
  end
end

C = 2*C/(n+1);
C(1) = C(1)/2;

