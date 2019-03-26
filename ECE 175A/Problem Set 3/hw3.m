clear all;
close all;

load data.mat;
load label.mat;

indeces0 = find(labelTrain == 0);
indeces1 = find(labelTrain == 1);
indeces2 = find(labelTrain == 2);
indeces3 = find(labelTrain == 3);
indeces4 = find(labelTrain == 4);
indeces5 = find(labelTrain == 5);
indeces6 = find(labelTrain == 6);
indeces7 = find(labelTrain == 7);
indeces8 = find(labelTrain == 8);
indeces9 = find(labelTrain == 9);

output0 = SampleMean(indeces0, imageTrain);
output1 = SampleMean(indeces1, imageTrain);
output2 = SampleMean(indeces2, imageTrain);
output3 = SampleMean(indeces3, imageTrain);
output4 = SampleMean(indeces4, imageTrain);
output5 = SampleMean(indeces5, imageTrain);
output6 = SampleMean(indeces6, imageTrain);
output7 = SampleMean(indeces7, imageTrain);
output8 = SampleMean(indeces8, imageTrain);
output9 = SampleMean(indeces9, imageTrain);

figure;
subplot(2, 5, 1);
imshow(uint8(output0));
title("Sample Mean of 0");
subplot(2, 5, 2);
imshow(uint8(output1));
title("Sample Mean of 1");
subplot(2, 5, 3);
imshow(uint8(output2));
title("Sample Mean of 2");
subplot(2, 5, 4);
imshow(uint8(output3));
title("Sample Mean of 3");
subplot(2, 5, 5);
imshow(uint8(output4));
title("Sample Mean of 4");
subplot(2, 5, 6);
imshow(uint8(output5));
title("Sample Mean of 5");
subplot(2, 5, 7);
imshow(uint8(output6));
title("Sample Mean of 6");
subplot(2, 5, 8);
imshow(uint8(output7));
title("Sample Mean of 7");
subplot(2, 5, 9);
imshow(uint8(output8));
title("Sample Mean of 8");
subplot(2, 5, 10);
imshow(uint8(output9));
title("Sample Mean of 9");

output1 = output1(:);
output2 = output2(:);
output3 = output3(:);
output4 = output4(:);
output0 = output0(:);
output5 = output5(:);
output6 = output6(:);
output7 = output7(:);
output8 = output8(:);
output9 = output9(:);
MasterOutput = horzcat(output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
FinalClassification = BayesDecision(imageTest, MasterOutput);

Comparison = FinalClassification == labelTest;
Error_Rate = (1 - sum(Comparison)/500) * 100;
class_Error_Rates = Class_Error(FinalClassification, labelTest);

Cov0 = CovarianceMatrix(indeces0, imageTrain);
Cov1 = CovarianceMatrix(indeces1, imageTrain);
Cov2 = CovarianceMatrix(indeces2, imageTrain);
Cov3 = CovarianceMatrix(indeces3, imageTrain);
Cov4 = CovarianceMatrix(indeces4, imageTrain);
Cov5 = CovarianceMatrix(indeces5, imageTrain);
Cov6 = CovarianceMatrix(indeces6, imageTrain);
Cov7 = CovarianceMatrix(indeces7, imageTrain);
Cov8 = CovarianceMatrix(indeces8, imageTrain);
Cov9 = CovarianceMatrix(indeces9, imageTrain);

figure;
subplot(2, 5, 1);
imshow(Cov0,[]);
title('covariance matrix of 0');
subplot(2, 5, 2);
imshow(Cov1,[]);
title('covariance matrix of 1');
subplot(2, 5, 3);
imshow(Cov2,[]);
title('covariance matrix of 2');
subplot(2, 5, 4);
imshow(Cov3,[]);
title('covariance matrix of 3');
subplot(2, 5, 5);
imshow(Cov4,[]);
title('covariance matrix of 4');
subplot(2, 5, 6);
imshow(Cov5,[]);
title('covariance matrix of 5');
subplot(2, 5, 7);
imshow(Cov6,[]);
title('covariance matrix of 6');
subplot(2, 5, 8);
imshow(Cov7,[]);
title('covariance matrix of 7');
subplot(2, 5, 9);
imshow(Cov8,[]);
title('covariance matrix of 8');
subplot(2, 5, 10);
imshow(Cov9,[]);
title('covariance matrix of 9');


function outputMean = SampleMean(anIndex, trainningImageSet) 
    [row, col] = size(anIndex);
    sumIndexValue = zeros(28, 28);
    for i = 1:row
        sumIndexValue = sumIndexValue + trainningImageSet(:,:,anIndex(i));
    end
    sumIndexValue = sumIndexValue / row;
    outputMean = sumIndexValue;
end

function BaesDecision = BayesDecision(Images, CompareOutput)
    BaesDecision = zeros(500, 1);
    for i = 1:500
        comparisonImage = Images(:, :, i);
        comparisonImage = comparisonImage(:);
        argMax = zeros(10, 1);
        for j = 1:10
            argMax(j) = -.5 * ((comparisonImage - CompareOutput(:, j))' * (comparisonImage - CompareOutput(:, j))) - .5 * 784 * log(2 * pi) + log(.1);
        end
        [~, maxIndex] = max(argMax);
        BaesDecision(i) = maxIndex - 1;
    end
end

function Class_Error_Rates = Class_Error(classifierLabels, testLabels)
    Class_Error_Rates = [];
    figure;
    hold on;
    for i = 0:9
        indeces = find(testLabels == i);
        num_Correct_Labels = sum(classifierLabels(indeces) == i);
        error = 1 - (num_Correct_Labels / size(indeces, 1));
        Class_Error_Rates = [Class_Error_Rates; error];
        bar(i, error);
    end
    xlabel("class");
    ylabel("error percentage");
    title("class vs error percentage bar graph");
    hold off;
end

function Output = CovarianceMatrix(anIndex, Images)
    [row, col] = size(anIndex);
    StackedImages = zeros(784, row);
    for i = 1:row
        anImage = Images(:, :, anIndex(i));
        anImage = anImage(:);
        StackedImages(:, i) = anImage;
    end
    
    Output = cov((StackedImages)');
end