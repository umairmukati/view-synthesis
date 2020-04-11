clearvars; clearvars -global; clc; close all;

addpath('Functions');

%InitParam();
InitParamSrinivasan();
tic
PrepareTrainingData();
toc

%PrepareTestData();


