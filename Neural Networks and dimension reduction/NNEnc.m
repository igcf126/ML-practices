function [y1] = NNEnc(x1)
%NNENC neural network simulation function.
%
% Auto-generated by MATLAB, 10-Apr-2022 21:50:16.
% 
% [y1] = NNEnc(x1) takes these arguments:
%   x = 8xQ matrix, input #1
% and returns:
%   y = 8xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [9;3;68;46;1613;8;70;1];
x1_step1.gain = [0.0531914893617021;0.4;0.00516795865633075;0.0108695652173913;0.000567054153671676;0.119047619047619;0.166666666666667;1];
x1_step1.ymin = -1;

% Layer 1
b1 = [0.41441591871731708885;0.51203829966844272015];
IW1_1 = [-0.0012652242415571296735 -0.00080067708486739757896 0.49614741372464338243 0.071463062728049933647 -0.19102910650046420193 -0.0010929003606585285975 -0.00015464263500368746447 -0.00064034776579914565436;-0.00021736244336239204391 -0.0001415101660095272973 0.076331765631642886638 0.010585017686963687708 -0.57051207238512158249 -0.00033272119606288111469 -6.9978888238115017771e-06 -7.4203943917844664927e-05];

% Layer 2
b2 = [-0.33995317613684344904;-0.0053932515333349907022;-0.46262421190872937693;-0.35100463287165728499;0.8282858203970084876;-0.072517244106456310582;-0.0054821152965618567474;-0.0243169795010675302];
LW2_1 = [-0.6922704739408036545 0.4806252202130280593;0.024284158809392403927 -0.032362464300040451326;1.9366322458716591637 -0.63634931329213317852;1.2542439434337846471 -0.55015035960243530155;0.28397648838857575404 -1.8471456823823191318;-0.13947277736628707623 0.0208237247837930739;-0.079269664902330783107 0.054298341470516195417;-0.0026891019335218637083 -0.015280200996170896499];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [0.0531914893617021;0.4;0.00516795865633075;0.0108695652173913;0.000567054153671676;0.119047619047619;0.166666666666667;1];
y1_step1.xoffset = [9;3;68;46;1613;8;70;1];

% ===== SIMULATION ========

% Dimensions
Q = size(x1,2); % samples

% Input 1
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = satlin_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = repmat(b2,1,Q) + LW2_1*a1;

% Output 1
y1 = mapminmax_reverse(a2,y1_step1);
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
  y = bsxfun(@minus,x,settings.xoffset);
  y = bsxfun(@times,y,settings.gain);
  y = bsxfun(@plus,y,settings.ymin);
end

% Linear Saturating Positive Transfer Function
function a = satlin_apply(n,~)
  a = max(0,min(1,n));
  a(isnan(n)) = nan;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
  x = bsxfun(@minus,y,settings.ymin);
  x = bsxfun(@rdivide,x,settings.gain);
  x = bsxfun(@plus,x,settings.xoffset);
end
