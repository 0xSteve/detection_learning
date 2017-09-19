%close all figures, clear the workspace, and clear the command window.
close all; clear; clc;
digitsC = digits;
digits(5);
depth = [0:0.1:70]; 
low_precision_depth = [0:1:69];

%Standard Normal Distribution%

%For the standard normal distribution we will assume the best depth will be at
%be 45m so we will use mu = 45m.

mu = 45;
sigma =5;
unimodalNormalVector = normpdf(depth, mu, sigma) / (2 * sigma);
unimodalNormalVector = round(unimodalNormalVector, 4);
sum(unimodalNormalVector)
figure
title('ACtual Distribution')
stem( unimodalNormalVector)
xlabel('Depth (dm)')
ylabel('PDF of the best link')
legend('mu = 45m, sigma = 5')

%Read in the distributions from the csv files and plot them.
learned_best = csvread('learnedBest.csv');
learned_dist = csvread('learnedDist.csv');
action1_p = csvread('Action1Probability.csv');
action0_p = csvread('Action0Probability.csv');
action2_p = csvread('Action2Probability.csv');

figure
stem(low_precision_depth, learned_best)
% title('Learned Best Depth')
xlabel('Depth (m)')
ylabel('Confidence')

figure
plot(action1_p)
title('Evolution of \alpha_1 Probability')
xlabel('Ensemble(n)')
ylabel('P(\alpha_1)')
% legend('best link at depth of 44m')
figure
plot((action0_p+ action2_p))
%title('Evolution of  Other Action Probability')
xlabel('Ensemble(n)')
ylabel('P(\alpha_0) + P(\alpha_2)')

figure
plot((action0_p))
% title('Evolution of  \alpha_0 Probability')
xlabel('Ensemble(n)')
ylabel('P(\alpha_0)')

figure
hold on
plot((action1_p))
plot((action0_p+ action2_p))
hold off
% title('Evolution of  \alpha_2 Probability')
xlabel('Ensemble(n)')
ylabel('Action Probability')
legend('P(\alpha_1)','P(\alpha_0) + P(\alpha_2)')
% figure
% 
% %title('Evolution of  Other Action Probability')
% xlabel('Ensemble(n)')
% ylabel('P(allstate')
