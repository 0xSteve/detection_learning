%Generate some test distributions to use for experimentation in the MaSt
%and PLar algorithms.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This script produces multiple probabilistic distributions to use for
%testing the automata's ability to find the maximum probability depth. The
%file outputs a CSV file with probability vectors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%to test their action update function.

%The depth will range from 0 meters to 70 meters where 0 is the sea
%surface.
%The granularity for testing will assume the automata can move in minimum
%incriments of 10cm. The depth will be used to evaluate the PDF of each
%type of distribution.

%close all figures, clear the workspace, and clear the command window.
close all; clear; clc;
digitsC = digits;
digits(5);
depth = [0:0.1:70]; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UNIMODAL PROBABILITY VECTORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Standard Normal Distribution%

%For the standard normal distribution we will assume the best depth will be at
%be 45m so we will use mu = 45m.

mu = 45;
sigma =5;
unimodalNormalVector = normpdf(depth, mu, sigma);
%unimodalNormalVector = normpdf([-3:0.1:3], 0, 5);
unimodalNormalVector = round(unimodalNormalVector, 4);
sum(unimodalNormalVector)
csvwrite('unimodalNormalVector.csv', unimodalNormalVector');
figure
stem([0:1:700], unimodalNormalVector)
xlabel('Depth (dm)')
ylabel('Probability')
% legend('\mu = 45m, \sigma = 5')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MULTIMODAL PROBABILITY VECTORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Bimodal Standard Normal distribution
mu1 = 15;
secondNormalVector = normpdf(depth, mu1, sigma+4);
csvwrite('otherNormalVector.csv', secondNormalVector');

bimodalVector = (unimodalNormalVector + secondNormalVector) / (2);
csvwrite('bimodalNormalVector.csv', bimodalVector');
figure
stem([0:1:700],bimodalVector)
xlabel('Depth (dm)')
ylabel('Probability')
% legend('\mu_1 = 15m  \mu_2=45m, \sigma = 5')

% %Bimodal
% mu = 20;
% uniformNormalVector2 = normpdf(depth,mu,(2 * sigma));
% bimodalpdf = (uniformNormalVector2 + unimodalNormalVector) / (2*sigma);
% 
% figure
% plot(depth, bimodalpdf)
% xlabel('Depth (m)')
% ylabel('P(Best Link)')
% digits(digitsC)
% disp('Completed successfully!')
% 
% 
