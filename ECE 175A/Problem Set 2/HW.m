load data.mat;
load label.mat;
trainImage = imageTrain;
trainLabel = labelTrain;
testImage = imageTest;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ;
testLabel = labelTest;
Label = nearest_Neighbor(trainImage, trainLabel, testImage, testLabel);
Comparison = Label == testLabel;
Error_Rate = (1 - sum(Comparison)/500) * 100;
class_Error_Rates = Class_Error(Label, testLabel);

function Labels = nearest_Neighbor(trainImage, trainLabel, testImage, testLabel)
    temp_Distance = [];
    Labels = [];
    for j = 1:500
        for i = 1:5000
           dist = (testImage(:, :, j) - trainImage(:, :, i)) .^ 2;
           %dist = (testImage - trainImage(:, :, i)) .^ 2;
           sum1 = sum(dist);
           actual_Sum = sum(sum1);
           temp_Distance = [temp_Distance sqrt(actual_Sum)];
        end

        [~, index] = min(temp_Distance);
        %disp(index);
        label = trainLabel(index);
        Labels = [Labels; label];
        temp_Distance = [];
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