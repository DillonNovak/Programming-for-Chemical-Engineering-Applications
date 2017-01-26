%%  ImprovedEuler.m
%   
%   Homework 7
%   22 March 2015
%   Author: Dillon Novak
%   Collaborators: Lizett Pink
%
%   This program is supposed to calculate and plot a function over a range
%   using the improved Euler method.
%


clear; clc;

%% set parameters
npoints = 300;
xmin = 0;
xmax = 180;
k=-0.028;
T = 60;
t1 = linspace(xmin,xmax,npoints);
dt = t1(2)-t1(1);

y1 = zeros(1,npoints);

%% for loop to calculate y1
% set initial y
y1(1) = 100;
for i=1:npoints-1
    
    y1(i+1)=(y1(i)*(1+k*dt/2)-T*k*dt)/(1-k*dt/2);

end

%% plot the function
plot(t1,y1,'b-')
xlabel('x');
ylabel('y(x)');
title(['Improved Euler Method Approximation with ' ,num2str(npoints),' points.']);