clear all;
close all;
load('data.mat');
load('label.mat');

% Part 1
ReshapedTrain = reshape(imageTrain, [784 5000])/255;
ReshapedTest = reshape(imageTest, [784 500])/255;
MeanTrain = mean(ReshapedTrain, 2);
MeanTest = mean(ReshapedTest, 2);
[EigMatTrain, EigValTrain] = Part1(ReshapedTrain, MeanTrain);
[EigMatTest, EigValTest] = Part1(ReshapedTest, MeanTest);
Part1_Plots = reshape(EigMatTrain(:, 1:10), [28 28 10]);


figure;
for i = 1:10
    subplot(2, 5, i);
    imshow(Part1_Plots(:, :, i), []);
    title(['Principal Component #' num2str(i)]);
end
sgtitle('Principal Components');
figure;
plot(EigValTrain);
title('Eigenvalues of Training Set');

% Part 1 for 5s
IndecesOf5 = find(labelTrain == 5);
ImageOf5s = ReshapedTrain(:, IndecesOf5);
MeanOf5s = mean(ImageOf5s, 2);
[EigMat5s, EigValTrain] = Part1(ImageOf5s, MeanOf5s);
Part1_5s_Plots = reshape(EigMat5s(:, 1:10), [28 28 10]);

figure;
for i = 1:10
    subplot(2, 5, i);
    imshow(Part1_5s_Plots(:, :, i), []);
    title(['Principal Component of digit 5 #' num2str(i)]);
end
sgtitle('Principal Components of digit 5');

% Part 2 a on paper
% d = 5
Indeces0 = find(labelTrain == 0);
Indeces1 = find(labelTrain == 1);
Indeces2 = find(labelTrain == 2);
Indeces3 = find(labelTrain == 3);
Indeces4 = find(labelTrain == 4);
Indeces5 = find(labelTrain == 5);
Indeces6 = find(labelTrain == 6);
Indeces7 = find(labelTrain == 7);
Indeces8 = find(labelTrain == 8);
Indeces9 = find(labelTrain == 9);
NewTestingData = PCAScaling(5, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(5, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(5) + cov(image0');
cov1 = 10 ^ -3 * eye(5) + cov(image1');
cov2 = 10 ^ -3 * eye(5) + cov(image2');
cov3 = 10 ^ -3 * eye(5) + cov(image3');
cov4 = 10 ^ -3 * eye(5) + cov(image4');
cov5 = 10 ^ -3 * eye(5) + cov(image5');
cov6 = 10 ^ -3 * eye(5) + cov(image6');
cov7 = 10 ^ -3 * eye(5) + cov(image7');
cov8 = 10 ^ -3 * eye(5) + cov(image8');
cov9 = 10 ^ -3 * eye(5) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 5);
output1 = SampleMean(Indeces1, NewTrainingData, 5);
output2 = SampleMean(Indeces2, NewTrainingData, 5);
output3 = SampleMean(Indeces3, NewTrainingData, 5);
output4 = SampleMean(Indeces4, NewTrainingData, 5);
output5 = SampleMean(Indeces5, NewTrainingData, 5);
output6 = SampleMean(Indeces6, NewTrainingData, 5);
output7 = SampleMean(Indeces7, NewTrainingData, 5);
output8 = SampleMean(Indeces8, NewTrainingData, 5);
output9 = SampleMean(Indeces9, NewTrainingData, 5);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision5 = Classification(NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison5 = BayesDecision5 == labelTest;
Error_Rate5 = (1 - sum(Comparison5)/500) * 100;

% d = 10
NewTestingData = PCAScaling(10, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(10, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(10) + cov(image0');
cov1 = 10 ^ -3 * eye(10) + cov(image1');
cov2 = 10 ^ -3 * eye(10) + cov(image2');
cov3 = 10 ^ -3 * eye(10) + cov(image3');
cov4 = 10 ^ -3 * eye(10) + cov(image4');
cov5 = 10 ^ -3 * eye(10) + cov(image5');
cov6 = 10 ^ -3 * eye(10) + cov(image6');
cov7 = 10 ^ -3 * eye(10) + cov(image7');
cov8 = 10 ^ -3 * eye(10) + cov(image8');
cov9 = 10 ^ -3 * eye(10) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 10);
output1 = SampleMean(Indeces1, NewTrainingData, 10);
output2 = SampleMean(Indeces2, NewTrainingData, 10);
output3 = SampleMean(Indeces3, NewTrainingData, 10);
output4 = SampleMean(Indeces4, NewTrainingData, 10);
output5 = SampleMean(Indeces5, NewTrainingData, 10);
output6 = SampleMean(Indeces6, NewTrainingData, 10);
output7 = SampleMean(Indeces7, NewTrainingData, 10);
output8 = SampleMean(Indeces8, NewTrainingData, 10);
output9 = SampleMean(Indeces9, NewTrainingData, 10);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision10 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison10 = BayesDecision10 == labelTest;
Error_Rate10 = (1 - sum(Comparison10)/500) * 100;

% d = 20
NewTestingData = PCAScaling(20, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(20, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(20) + cov(image0');
cov1 = 10 ^ -3 * eye(20) + cov(image1');
cov2 = 10 ^ -3 * eye(20) + cov(image2');
cov3 = 10 ^ -3 * eye(20) + cov(image3');
cov4 = 10 ^ -3 * eye(20) + cov(image4');
cov5 = 10 ^ -3 * eye(20) + cov(image5');
cov6 = 10 ^ -3 * eye(20) + cov(image6');
cov7 = 10 ^ -3 * eye(20) + cov(image7');
cov8 = 10 ^ -3 * eye(20) + cov(image8');
cov9 = 10 ^ -3 * eye(20) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 20);
output1 = SampleMean(Indeces1, NewTrainingData, 20);
output2 = SampleMean(Indeces2, NewTrainingData, 20);
output3 = SampleMean(Indeces3, NewTrainingData, 20);
output4 = SampleMean(Indeces4, NewTrainingData, 20);
output5 = SampleMean(Indeces5, NewTrainingData, 20);
output6 = SampleMean(Indeces6, NewTrainingData, 20);
output7 = SampleMean(Indeces7, NewTrainingData, 20);
output8 = SampleMean(Indeces8, NewTrainingData, 20);
output9 = SampleMean(Indeces9, NewTrainingData, 20);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision20 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison20 = BayesDecision20 == labelTest;
Error_Rate20 = (1 - sum(Comparison20)/500) * 100;

% d = 30
NewTestingData = PCAScaling(30, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(30, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(30) + cov(image0');
cov1 = 10 ^ -3 * eye(30) + cov(image1');
cov2 = 10 ^ -3 * eye(30) + cov(image2');
cov3 = 10 ^ -3 * eye(30) + cov(image3');
cov4 = 10 ^ -3 * eye(30) + cov(image4');
cov5 = 10 ^ -3 * eye(30) + cov(image5');
cov6 = 10 ^ -3 * eye(30) + cov(image6');
cov7 = 10 ^ -3 * eye(30) + cov(image7');
cov8 = 10 ^ -3 * eye(30) + cov(image8');
cov9 = 10 ^ -3 * eye(30) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 30);
output1 = SampleMean(Indeces1, NewTrainingData, 30);
output2 = SampleMean(Indeces2, NewTrainingData, 30);
output3 = SampleMean(Indeces3, NewTrainingData, 30);
output4 = SampleMean(Indeces4, NewTrainingData, 30);
output5 = SampleMean(Indeces5, NewTrainingData, 30);
output6 = SampleMean(Indeces6, NewTrainingData, 30);
output7 = SampleMean(Indeces7, NewTrainingData, 30);
output8 = SampleMean(Indeces8, NewTrainingData, 30);
output9 = SampleMean(Indeces9, NewTrainingData, 30);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision30 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison30 = BayesDecision30 == labelTest;
Error_Rate30 = (1 - sum(Comparison30)/500) * 100;

% d = 40
NewTestingData = PCAScaling(40, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(40, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(40) + cov(image0');
cov1 = 10 ^ -3 * eye(40) + cov(image1');
cov2 = 10 ^ -3 * eye(40) + cov(image2');
cov3 = 10 ^ -3 * eye(40) + cov(image3');
cov4 = 10 ^ -3 * eye(40) + cov(image4');
cov5 = 10 ^ -3 * eye(40) + cov(image5');
cov6 = 10 ^ -3 * eye(40) + cov(image6');
cov7 = 10 ^ -3 * eye(40) + cov(image7');
cov8 = 10 ^ -3 * eye(40) + cov(image8');
cov9 = 10 ^ -3 * eye(40) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 40);
output1 = SampleMean(Indeces1, NewTrainingData, 40);
output2 = SampleMean(Indeces2, NewTrainingData, 40);
output3 = SampleMean(Indeces3, NewTrainingData, 40);
output4 = SampleMean(Indeces4, NewTrainingData, 40);
output5 = SampleMean(Indeces5, NewTrainingData, 40);
output6 = SampleMean(Indeces6, NewTrainingData, 40);
output7 = SampleMean(Indeces7, NewTrainingData, 40);
output8 = SampleMean(Indeces8, NewTrainingData, 40);
output9 = SampleMean(Indeces9, NewTrainingData, 40);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision40 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison40 = BayesDecision40 == labelTest;
Error_Rate40 = (1 - sum(Comparison40)/500) * 100;

% d = 60
NewTestingData = PCAScaling(60, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(60, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(60) + cov(image0');
cov1 = 10 ^ -3 * eye(60) + cov(image1');
cov2 = 10 ^ -3 * eye(60) + cov(image2');
cov3 = 10 ^ -3 * eye(60) + cov(image3');
cov4 = 10 ^ -3 * eye(60) + cov(image4');
cov5 = 10 ^ -3 * eye(60) + cov(image5');
cov6 = 10 ^ -3 * eye(60) + cov(image6');
cov7 = 10 ^ -3 * eye(60) + cov(image7');
cov8 = 10 ^ -3 * eye(60) + cov(image8');
cov9 = 10 ^ -3 * eye(60) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 60);
output1 = SampleMean(Indeces1, NewTrainingData, 60);
output2 = SampleMean(Indeces2, NewTrainingData, 60);
output3 = SampleMean(Indeces3, NewTrainingData, 60);
output4 = SampleMean(Indeces4, NewTrainingData, 60);
output5 = SampleMean(Indeces5, NewTrainingData, 60);
output6 = SampleMean(Indeces6, NewTrainingData, 60);
output7 = SampleMean(Indeces7, NewTrainingData, 60);
output8 = SampleMean(Indeces8, NewTrainingData, 60);
output9 = SampleMean(Indeces9, NewTrainingData, 60);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision60 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison60 = BayesDecision60 == labelTest;
Error_Rate60 = (1 - sum(Comparison60)/500) * 100;

% d = 90
NewTestingData = PCAScaling(90, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(90, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(90) + cov(image0');
cov1 = 10 ^ -3 * eye(90) + cov(image1');
cov2 = 10 ^ -3 * eye(90) + cov(image2');
cov3 = 10 ^ -3 * eye(90) + cov(image3');
cov4 = 10 ^ -3 * eye(90) + cov(image4');
cov5 = 10 ^ -3 * eye(90) + cov(image5');
cov6 = 10 ^ -3 * eye(90) + cov(image6');
cov7 = 10 ^ -3 * eye(90) + cov(image7');
cov8 = 10 ^ -3 * eye(90) + cov(image8');
cov9 = 10 ^ -3 * eye(90) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 90);
output1 = SampleMean(Indeces1, NewTrainingData, 90);
output2 = SampleMean(Indeces2, NewTrainingData, 90);
output3 = SampleMean(Indeces3, NewTrainingData, 90);
output4 = SampleMean(Indeces4, NewTrainingData, 90);
output5 = SampleMean(Indeces5, NewTrainingData, 90);
output6 = SampleMean(Indeces6, NewTrainingData, 90);
output7 = SampleMean(Indeces7, NewTrainingData, 90);
output8 = SampleMean(Indeces8, NewTrainingData, 90);
output9 = SampleMean(Indeces9, NewTrainingData, 90);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision90 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison90 = BayesDecision90 == labelTest;
Error_Rate90 = (1 - sum(Comparison90)/500) * 100;

% d = 130
NewTestingData = PCAScaling(130, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(130, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(130) + cov(image0');
cov1 = 10 ^ -3 * eye(130) + cov(image1');
cov2 = 10 ^ -3 * eye(130) + cov(image2');
cov3 = 10 ^ -3 * eye(130) + cov(image3');
cov4 = 10 ^ -3 * eye(130) + cov(image4');
cov5 = 10 ^ -3 * eye(130) + cov(image5');
cov6 = 10 ^ -3 * eye(130) + cov(image6');
cov7 = 10 ^ -3 * eye(130) + cov(image7');
cov8 = 10 ^ -3 * eye(130) + cov(image8');
cov9 = 10 ^ -3 * eye(130) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 130);
output1 = SampleMean(Indeces1, NewTrainingData, 130);
output2 = SampleMean(Indeces2, NewTrainingData, 130);
output3 = SampleMean(Indeces3, NewTrainingData, 130);
output4 = SampleMean(Indeces4, NewTrainingData, 130);
output5 = SampleMean(Indeces5, NewTrainingData, 130);
output6 = SampleMean(Indeces6, NewTrainingData, 130);
output7 = SampleMean(Indeces7, NewTrainingData, 130);
output8 = SampleMean(Indeces8, NewTrainingData, 130);
output9 = SampleMean(Indeces9, NewTrainingData, 130);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision130 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison130 = BayesDecision130 == labelTest;
Error_Rate130 = (1 - sum(Comparison130)/500) * 100;

% d = 180
NewTestingData = PCAScaling(180, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(180, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(180) + cov(image0');
cov1 = 10 ^ -3 * eye(180) + cov(image1');
cov2 = 10 ^ -3 * eye(180) + cov(image2');
cov3 = 10 ^ -3 * eye(180) + cov(image3');
cov4 = 10 ^ -3 * eye(180) + cov(image4');
cov5 = 10 ^ -3 * eye(180) + cov(image5');
cov6 = 10 ^ -3 * eye(180) + cov(image6');
cov7 = 10 ^ -3 * eye(180) + cov(image7');
cov8 = 10 ^ -3 * eye(180) + cov(image8');
cov9 = 10 ^ -3 * eye(180) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 180);
output1 = SampleMean(Indeces1, NewTrainingData, 180);
output2 = SampleMean(Indeces2, NewTrainingData, 180);
output3 = SampleMean(Indeces3, NewTrainingData, 180);
output4 = SampleMean(Indeces4, NewTrainingData, 180);
output5 = SampleMean(Indeces5, NewTrainingData, 180);
output6 = SampleMean(Indeces6, NewTrainingData, 180);
output7 = SampleMean(Indeces7, NewTrainingData, 180);
output8 = SampleMean(Indeces8, NewTrainingData, 180);
output9 = SampleMean(Indeces9, NewTrainingData, 180);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision180 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison180 = BayesDecision180 == labelTest;
Error_Rate180 = (1 - sum(Comparison180)/500) * 100;

% d = 250
NewTestingData = PCAScaling(250, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(250, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(250) + cov(image0');
cov1 = 10 ^ -3 * eye(250) + cov(image1');
cov2 = 10 ^ -3 * eye(250) + cov(image2');
cov3 = 10 ^ -3 * eye(250) + cov(image3');
cov4 = 10 ^ -3 * eye(250) + cov(image4');
cov5 = 10 ^ -3 * eye(250) + cov(image5');
cov6 = 10 ^ -3 * eye(250) + cov(image6');
cov7 = 10 ^ -3 * eye(250) + cov(image7');
cov8 = 10 ^ -3 * eye(250) + cov(image8');
cov9 = 10 ^ -3 * eye(250) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 250);
output1 = SampleMean(Indeces1, NewTrainingData, 250);
output2 = SampleMean(Indeces2, NewTrainingData, 250);
output3 = SampleMean(Indeces3, NewTrainingData, 250);
output4 = SampleMean(Indeces4, NewTrainingData, 250);
output5 = SampleMean(Indeces5, NewTrainingData, 250);
output6 = SampleMean(Indeces6, NewTrainingData, 250);
output7 = SampleMean(Indeces7, NewTrainingData, 250);
output8 = SampleMean(Indeces8, NewTrainingData, 250);
output9 = SampleMean(Indeces9, NewTrainingData, 250);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision250 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison250 = BayesDecision250 == labelTest;
Error_Rate250 = (1 - sum(Comparison250)/500) * 100;

% % d = 350
NewTestingData = PCAScaling(350, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling(350, EigMatTrain, ReshapedTrain, MeanTrain);
image0 = NewTrainingData(:, Indeces0);
image1 = NewTrainingData(:, Indeces1);
image2 = NewTrainingData(:, Indeces2);
image3 = NewTrainingData(:, Indeces3);
image4 = NewTrainingData(:, Indeces4);
image5 = NewTrainingData(:, Indeces5);
image6 = NewTrainingData(:, Indeces6);
image7 = NewTrainingData(:, Indeces7);
image8 = NewTrainingData(:, Indeces8);
image9 = NewTrainingData(:, Indeces9);
cov0 = 10 ^ -3 * eye(350) + cov(image0');
cov1 = 10 ^ -3 * eye(350) + cov(image1');
cov2 = 10 ^ -3 * eye(350) + cov(image2');
cov3 = 10 ^ -3 * eye(350) + cov(image3');
cov4 = 10 ^ -3 * eye(350) + cov(image4');
cov5 = 10 ^ -3 * eye(350) + cov(image5');
cov6 = 10 ^ -3 * eye(350) + cov(image6');
cov7 = 10 ^ -3 * eye(350) + cov(image7');
cov8 = 10 ^ -3 * eye(350) + cov(image8');
cov9 = 10 ^ -3 * eye(350) + cov(image9');
output0 = SampleMean(Indeces0, NewTrainingData, 350);
output1 = SampleMean(Indeces1, NewTrainingData, 350);
output2 = SampleMean(Indeces2, NewTrainingData, 350);
output3 = SampleMean(Indeces3, NewTrainingData, 350);
output4 = SampleMean(Indeces4, NewTrainingData, 350);
output5 = SampleMean(Indeces5, NewTrainingData, 350);
output6 = SampleMean(Indeces6, NewTrainingData, 350);
output7 = SampleMean(Indeces7, NewTrainingData, 350);
output8 = SampleMean(Indeces8, NewTrainingData, 350);
output9 = SampleMean(Indeces9, NewTrainingData, 350);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
BayesDecision350 = Classification( NewTestingData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7,...
    cov8, cov9, MasterOutput);
Comparison350 = BayesDecision350 == labelTest;
Error_Rate350 = (1 - sum(Comparison350)/500) * 100;

% graphing the error rates
errors = [Error_Rate5 Error_Rate10 Error_Rate20 Error_Rate30 Error_Rate40 Error_Rate60 Error_Rate90 Error_Rate130 Error_Rate180 ...
    Error_Rate250 Error_Rate350];
constants = [5 10 20 30 40 60 90 130 180 250 350];
figure;
bar(constants, errors);
title("Error Rates given d");

% part 3
NewTestingData = PCAScaling2(40, EigMatTrain, ReshapedTest, MeanTrain);
NewTrainingData = PCAScaling2(40, EigMatTrain, ReshapedTrain, MeanTrain);
image5 = NewTrainingData(:, Indeces5);
cov5 = 10 ^ -3 * eye(744) + cov(image5');
output5 = SampleMean(Indeces5, NewTrainingData, 744);
LeastLike5Index = Classification2(NewTestingData, cov5, output5);
figure;
imshow(imageTest(:, :, LeastLike5Index), []);

function ImageIndex = Classification2(TestData, cov, mean)
    BaesDecision = zeros(500, 1);
    for i = 1:500
        comparisonImage = TestData(:, i);
        BaesDecision(i) = mvnpdf(comparisonImage, mean, cov);
    end
    [~, ImageIndex] = min(BaesDecision);
end

function BaesDecision = Classification(TestData, cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7, cov8, cov9, CompareOutput)
    BaesDecision = zeros(500, 1);
    for i = 1:500
        comparisonImage = TestData(:, i);
        argMax = zeros(10, 1);
        argMax(1) = mvnpdf(comparisonImage, CompareOutput(:, 1), cov0);
        argMax(2) = mvnpdf(comparisonImage, CompareOutput(:, 2), cov1);
        argMax(3) = mvnpdf(comparisonImage, CompareOutput(:, 3), cov2);
        argMax(4) = mvnpdf(comparisonImage, CompareOutput(:, 4), cov3);
        argMax(5) = mvnpdf(comparisonImage, CompareOutput(:, 5), cov4);
        argMax(6) = mvnpdf(comparisonImage, CompareOutput(:, 6), cov5);
        argMax(7) = mvnpdf(comparisonImage, CompareOutput(:, 7), cov6);
        argMax(8) = mvnpdf(comparisonImage, CompareOutput(:, 8), cov7);
        argMax(9) = mvnpdf(comparisonImage, CompareOutput(:, 9), cov8);
        argMax(10) = mvnpdf(comparisonImage, CompareOutput(:, 10), cov9);
        [~, maxIndex] = max(argMax);
        BaesDecision(i) = maxIndex - 1;
    end
end

function outputMean = SampleMean(anIndex, trainningImageSet, dim) 
    [row, ~] = size(anIndex);
    sumIndexValue = zeros(dim, 1);
    for i = 1:row
        sumIndexValue(:, 1) = sumIndexValue(:, 1) + trainningImageSet(:, anIndex(i));
    end
    sumIndexValue = sumIndexValue / row;
    outputMean = sumIndexValue;
end


function New_Data = PCAScaling(Dim, EigVec, Data, Mean)
    eigVecs = EigVec(:, 1:Dim);
    New_Data = eigVecs' * (Data - Mean);
end

function New_Data = PCAScaling2(Dim, EigVec, Data, Mean)
    eigVecs = EigVec(:, Dim+1:784);
    New_Data = eigVecs' * (Data - Mean);
end

function [MaxEigVec, MaxEigVal] = Part1(Data, Mean)
    % Calculate Covariances
    [row, col] = size(Data);
    CovTrain = zeros(784, 1);
    for i=1:col
        CovTrain = CovTrain + ((Data(:, i) - Mean) * (Data(:, i) - Mean)');
    end

    % Finding Eigenvalue and EigenVectors
    [trainEigVec, trainEigVal] = eig(CovTrain);
    trainEigVal = diag(trainEigVal);

    % Finding max Eigenvalues and Eigenvectors for testing set and trainning set 
    [MaxEigVal, MaxIndecesTrain] = sort(trainEigVal, 'descend');
    MaxEigVec = trainEigVec(:, MaxIndecesTrain);
end