clear all;
close all;
load data.mat;
load label.mat;
% 
% %part 1a
addpath('libsvm-3.23/matlab/');
Index6 = find(labelTrain == 6);
Index8 = find(labelTrain == 8);
Index68 = [Index6; Index8];
TestIndex6 = find(labelTest == 6);
TestIndex8 = find(labelTest == 8);
TestIndex68 = [TestIndex6; TestIndex8];
TestImage68 = imageTest(:, :, TestIndex68);
TestImage68 = reshape(TestImage68, [784 83]) / 255;
TestLabel68 = labelTest(TestIndex68);

Image68 = imageTrain(:, :, Index68);
Image68 = reshape(Image68, [784 963]) / 255;
Label68 = labelTrain(Index68);
svmopts = ['-t 0 -c 0.0625'];
TrainModel = svmtrain(Label68, Image68', svmopts);
[Yout, Accuracy, Yext] = svmpredict(TestLabel68, TestImage68', TrainModel, '');
% 
% %part 1b
% svCoeff6 = TrainModel.sv_coef(1:47)';
% [svCoeff6, svCoeff6Indeces] = sort(svCoeff6);
% SVs = full(TrainModel.SVs)';
% SV6s = SVs(:, svCoeff6Indeces(1:5));
% Closest6s = reshape(SV6s, 28, 28, 5);
% for i = 1:5
%     subplot(1, 5, i);
%     imshow(Closest6s(:, :, i), []);
%     title(['closest ' num2str(i) 'th SV to 6']);
% end
% 
% svCoeff8 = TrainModel.sv_coef(48:95)';
% [svCoeff8, svCoeff8Indeces] = sort(svCoeff8);
% svCoeff8Indeces = svCoeff8Indeces + 47;
% svCoeff8Indeces = fliplr(svCoeff8Indeces);
% SV8s = SVs(:, svCoeff8Indeces(1:5));
% Closest8s = reshape(SV8s, 28, 28, 5);
% figure;
% for i = 1:5
%     subplot(1, 5, i);
%     imshow(Closest8s(:, :, i), []);
%     title(['closest ' num2str(i) 'th SV to 8']);
% end
% 
% %part 1c
% figure;
% rho = TrainModel.rho;
% W = SVs * (TrainModel.sv_coef);
% ImageW = reshape(W, 28, 28);
% imshow(ImageW, []);
% 
% %part 1d
% distances=zeros(1,83);
% for i = 1:83
%     distances(i)=abs(W' * TestImage68(:,i) - rho) / norm(W);
% end
% 
% figure;
% histogram(distances,10);

% %part 1f
% distance6s = distances(1:43);
% [distance6s, indeces6s] = sort(distance6s);
% TestImage68 = reshape(TestImage68, [28 28 83]);
% distance8s = distances(44:83);
% [distance8s, indeces8s] = sort(distance8s);
% indeces8s = indeces8s + 43;
% figure;
% for i = 1:5
%     subplot(1, 5, i);
%     imshow(TestImage68(:, :, indeces6s(i)), []);
%     title(['closest ' num2str(i) ' 6 to boundary']);
% end
% 
% figure;
% for i = 1:5
%     subplot(1, 5, i);
%     imshow(TestImage68(:, :, indeces8s(i)), []);
%     title(['closest ' num2str(i) ' 8 to boundary']);
% end

%part 1e
% indeces8s = fliplr(indeces8s);
% indeces6s = fliplr(indeces6s);
% figure;
% for i = 1:5
%     subplot(1, 5, i);
%     imshow(TestImage68(:, :, indeces6s(i)), []);
%     title(['furthest ' num2str(i) ' 6 to boundary']);
% end
% 
% figure;
% for i = 1:5
%     subplot(1, 5, i);
%     imshow(TestImage68(:, :, indeces8s(i)), []);
%     title(['furthest ' num2str(i) ' 8 to boundary']);
% end

%part 2a
% C = [2^-3, 2^-1, 2^1, 2^3, 2^5, 2^7, 2^9, 2^11];
% Gamma = [2^-11, 2^-9, 2^-7, 2^-5, 2^-3, 2^-1];
% 
% sizeC = length(C);
% sizeGamma = length(Gamma);
% Errors = zeros(sizeC, sizeGamma);
% imageTrain = reshape(rescale(imageTrain, 0, 1), [784 5000]);
% imageTrain = imageTrain';
% imageTest = reshape(rescale(imageTest, 0, 1), [784 500]);
% imageTest = imageTest';
% for i = 1:sizeC
%     for j = 1:sizeGamma
%         formatSpec = '-t 2 -c %f -g %f -v 2';
%         cost = C(i);
%         gamma = Gamma(j);
%         string = sprintf(formatSpec, cost, gamma);
%         model2 = svmtrain(labelTrain, imageTrain, [string]);
%         Errors(i,j) = model2;
%     end
% end
% 
% formatSpec = '-t 2 -c %f -g %f';
% cost=2^1;
% gamma=2 ^-5;
% str = sprintf(formatSpec,cost,gamma);
% 
% model2b = svmtrain(labelTrain, imageTrain, [str]);
% [Yout, Acc, Yext] = svmpredict(labelTest, imageTest, model2b, '');