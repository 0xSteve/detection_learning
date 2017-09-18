%close all figures, clear the workspace, and clear the command window.
close all; clear; clc;
digitsC = digits;
digits(5);
depth = [0:0.1:70]; 

%Read in the distributions from the csv files and plot them.
bimodal = csvread('bimodalNormalVector.csv');

figure
stem(bimodal)
% title('Learned Best Depth')
xlabel('Depth (m)')
ylabel('Probability')

%Read in the distributions from the csv files and plot them.
linearDistEst = csvread('maxEstimate.csv');
linearDistEst =linearDistEst;
figure
stem(linearDistEst(:,1))
% title('Learned Best Depth')
xlabel('Depth (m)')
ylabel('Probability')
