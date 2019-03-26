%initializing random number between 0 to 1 28x28 array and part 1
r0 = rand(28);
r1 = rand(28);
r2 = rand(28);
r3 = rand(28);
r4 = rand(28);
r5 = rand(28);
r6 = rand(28);
r7 = rand(28);
r8 = rand(28);
r9 = rand(28);
r0 = r0(:);
r1 = r1(:);
r2 = r2(:);
r3 = r3(:);
r4 = r4(:);
r5 = r5(:);
r6 = r6(:);
r7 = r7(:);
r8 = r8(:);
r9 = r9(:);
[r0, r1, r2, r3, r4, r5, r6, r7, r8, r9] = k_Means(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, imageTrain);
r0 = reshape(r0, [28, 28]);
r1 = reshape(r1, [28, 28]);
r2 = reshape(r2, [28, 28]);
r3 = reshape(r3, [28, 28]);
r4 = reshape(r4, [28, 28]);
r5 = reshape(r5, [28, 28]);
r6 = reshape(r6, [28, 28]);
r7 = reshape(r7, [28, 28]);
r8 = reshape(r8, [28, 28]);
r9 = reshape(r9, [28, 28]);
imshow(uint8(r0));
imshow(uint8(r1));
imshow(uint8(r2));
imshow(uint8(r3));
imshow(uint8(r4));
imshow(uint8(r5));
imshow(uint8(r6));
imshow(uint8(r7));
imshow(uint8(r8));
imshow(uint8(r9));

%part 2
randomints = randi([1 5000], 1, 10);
image1 = imageTrain(:,:, randomints(1));
image2 = imageTrain(:,:, randomints(2));
image3 = imageTrain(:,:, randomints(3));
image4 = imageTrain(:,:, randomints(4));
image5 = imageTrain(:,:, randomints(5));
image6 = imageTrain(:,:, randomints(6));
image7 = imageTrain(:,:, randomints(7));
image8 = imageTrain(:,:, randomints(8));
image9 = imageTrain(:,:, randomints(9));
image10 = imageTrain(:,:, randomints(10));
image1 = image1(:);
image2 = image2(:);
image3 = image3(:);
image4 = image4(:);
image5 = image5(:);
image6 = image6(:);
image7 = image7(:);
image8 = image8(:);
image9 = image9(:);
image10 = image10(:);
[image1, image2, image3, image4, image5, image6, image7, image8, image9, image10] = k_Means(image1, image2, image3, ...
image4, image5, image6, image7, image8, image9, image10, imageTrain);
image1 = reshape(image1, [28, 28]);
image2 = reshape(image2, [28, 28]);
image3 = reshape(image3, [28, 28]);
image4 = reshape(image4, [28, 28]);
image5 = reshape(image5, [28, 28]);
image6 = reshape(image6, [28, 28]);
image7 = reshape(image7, [28, 28]);
image8 = reshape(image8, [28, 28]);
image9 = reshape(image9, [28, 28]);
image10 = reshape(image10, [28, 28]);
imshow(uint8(image1));
imshow(uint8(image2));
imshow(uint8(image3));
imshow(uint8(image4));
imshow(uint8(image5));
imshow(uint8(image6));
imshow(uint8(image7));
imshow(uint8(image8));
imshow(uint8(image9));
imshow(uint8(image10));

%part 3
image1 = image1(:);
image2 = image2(:);
image3 = image3(:);
image4 = image4(:);
image5 = image5(:);
image6 = image6(:);
image7 = image7(:);
image8 = image8(:);
image9 = image9(:);
image10 = image10(:);
MasterOutput = horzcat(image1, image2, image3, image4, image5, image6, image7, image8, image9, image10);
FinalClassification = BayesDecision(imageTest, MasterOutput);

Comparison = FinalClassification == labelTest;
Error_Rate = (1 - sum(Comparison)/500) * 100;
class_Error_Rates = Class_Error(FinalClassification, labelTest);

%part 4
randomints2 = randi([1 5000], 1, 10);
image11 = imageTrain(:,:, randomints2(1));
image12 = imageTrain(:,:, randomints2(2));
image13 = imageTrain(:,:, randomints2(3));
image14 = imageTrain(:,:, randomints2(4));
image15 = imageTrain(:,:, randomints2(5));
image16 = imageTrain(:,:, randomints2(6));
image17 = imageTrain(:,:, randomints2(7));
image18 = imageTrain(:,:, randomints2(8));
image19 = imageTrain(:,:, randomints2(9));
image20 = imageTrain(:,:, randomints2(10));
image11 = image11(:);
image12 = image12(:);
image13 = image13(:);
image14 = image14(:);
image15 = image15(:);
image16 = image16(:);
image17 = image17(:);
image18 = image18(:);
image19 = image19(:);
image20 = image20(:);
[image11, image12, image13, image14, image15, image16, image17, image18, image19, image20] = k_Means(image11, image12, image13, ...
image14, image15, image16, image17, image18, image19, image20, imageTrain);
image11 = reshape(image11, [28, 28]);
image12 = reshape(image12, [28, 28]);
image13 = reshape(image13, [28, 28]);
image14 = reshape(image14, [28, 28]);
image15 = reshape(image15, [28, 28]);
image16 = reshape(image16, [28, 28]);
image17 = reshape(image17, [28, 28]);
image18 = reshape(image18, [28, 28]);
image19 = reshape(image19, [28, 28]);
image20 = reshape(image20, [28, 28]);
imshow(uint8(image11));
imshow(uint8(image12));
imshow(uint8(image13));
imshow(uint8(image14));
imshow(uint8(image15));
imshow(uint8(image16));
imshow(uint8(image17));
imshow(uint8(image18));
imshow(uint8(image19));
imshow(uint8(image20));
image11 = image11(:);
image12 = image12(:);
image13 = image13(:);
image14 = image14(:);
image15 = image15(:);
image16 = image16(:);
image17 = image17(:);
image18 = image18(:);
image19 = image19(:);
image20 = image20(:);
MasterOutput2 = horzcat(image11, image12, image13, image14, image15, image16, image17, image18, image19, image20);
FinalClassification2 = BayesDecision(imageTest, MasterOutput2);

Comparison2 = FinalClassification2 == labelTest;
Error_Rate2 = (1 - sum(Comparison2)/500) * 100;
class_Error_Rates2 = Class_Error(FinalClassification2, labelTest);


function [R0, R1, R2, R3, R4, R5, R6, R7, R8, R9] = k_Means(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, testImage)
    count0 = 0;
    count1 = 0;
    count2 = 0;
    count3 = 0;
    count4 = 0;
    count5 = 0;
    count6 = 0;
    count7 = 0;
    count8 = 0;
    count9 = 0;
    while true
        image_r0 = [];
        image_r1 = [];
        image_r2 = [];
        image_r3 = [];
        image_r4 = [];
        image_r5 = [];
        image_r6 = [];
        image_r7 = [];
        image_r8 = [];
        image_r9 = [];
        for i = 1:5000
            Image = testImage(:, :, i);
            Image = Image(:);
            diff0 = abs(r0 - Image);
            diff1 = abs(r1 - Image);
            diff2 = abs(r2 - Image);
            diff3 = abs(r3 - Image);
            diff4 = abs(r4 - Image);
            diff5 = abs(r5 - Image);
            diff6 = abs(r6 - Image);
            diff7 = abs(r7 - Image);
            diff8 = abs(r8 - Image);
            diff9 = abs(r9 - Image);
            diff0 = sum(diff0);
            diff1 = sum(diff1);
            diff2 = sum(diff2);
            diff3 = sum(diff3);
            diff4 = sum(diff4);
            diff5 = sum(diff5);
            diff6 = sum(diff6);
            diff7 = sum(diff7);
            diff8 = sum(diff8);
            diff9 = sum(diff9);
            differences = [diff0; diff1; diff2; diff3; diff4; diff5; diff6; diff7; diff8; diff9];
            [~, index] = min(differences);
            if (index == 1)
                image_r0 = [image_r0 Image];
            elseif (index == 2)
                image_r1 = [image_r1 Image];
            elseif (index == 3)
                image_r2 = [image_r2 Image];
            elseif (index == 4)
                image_r3 = [image_r3 Image];
            elseif (index == 5)
                image_r4 = [image_r4 Image];
            elseif (index == 6)
                image_r5 = [image_r5 Image];
            elseif (index == 7)
                image_r6 = [image_r6 Image];
            elseif (index == 8)
                image_r7 = [image_r7 Image];
            elseif (index == 9)
                image_r8 = [image_r8 Image];
            else
                image_r9 = [image_r9 Image];
            end
        end
        
        if (abs(length(image_r0) - count0) <= 10 && abs(length(image_r1) - count1) <= 10 && abs(length(image_r2) - count2) <= 10 && ...
            abs(length(image_r3) - count3) <= 10 && abs(length(image_r4) - count4) <= 10 && abs(length(image_r5) - count5) <= 10 && ...
            abs(length(image_r6) - count6) <= 10 && abs(length(image_r7) - count7) <= 10 && abs(length(image_r8) - count8) <= 10 && ...
            abs(length(image_r9) - count9) <= 10)
            R0 = r0;
            R1 = r1;
            R2 = r2;
            R3 = r3;
            R4 = r4;
            R5 = r5;
            R6 = r6;
            R7 = r7;
            R8 = r8;
            R9 = r9;
            break;
        else
            count0 = length(image_r0);
            count1 = length(image_r1);
            count2 = length(image_r2);
            count3 = length(image_r3);
            count4 = length(image_r4);
            count5 = length(image_r5);
            count6 = length(image_r6);
            count7 = length(image_r7);
            count8 = length(image_r8);
            count9 = length(image_r9);
            r0 = sum(image_r0, 2) / size(image_r0, 2);
            r1 = sum(image_r1, 2) / size(image_r1, 2);
            r2 = sum(image_r2, 2) / size(image_r2, 2);
            r3 = sum(image_r3, 2) / size(image_r3, 2);
            r4 = sum(image_r4, 2) / size(image_r4, 2);
            r5 = sum(image_r5, 2) / size(image_r5, 2);
            r6 = sum(image_r6, 2) / size(image_r6, 2);
            r7 = sum(image_r7, 2) / size(image_r7, 2);
            r8 = sum(image_r8, 2) / size(image_r8, 2);
            r9 = sum(image_r9, 2) / size(image_r9, 2);
        end
    end
end

function BaesDecision = BayesDecision(Images, CompareOutput)
    BaesDecision = zeros(500, 1);
    labels = [0; 3; 9; 9; 4; 4; 6; 8; 5; 1];
    for i = 1:500
        comparisonImage = Images(:, :, i);
        comparisonImage = comparisonImage(:);
        argMax = zeros(10, 1);
        for j = 1:10
            argMax(j) = -.5 * ((comparisonImage - CompareOutput(:, j))' * (comparisonImage - CompareOutput(:, j))) - .5 * 784 * log(2 * pi) + log(.1);
        end
        [~, maxIndex] = max(argMax);
        BaesDecision(i) = labels(maxIndex);
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
    
    
    