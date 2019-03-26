%part a
x = sampletrain(:);
y = sampletest(:);
x = double(x);
y = double(y);
a = inv(x' * x) * x' * y;

%part b
trainImage = imageTrain;
trainLabel = labelTrain;
testImage = imageTestNew;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ;
testLabel = labelTestNew;
Least_Squares_Label = least_Squares_Nearest_Neighbor(trainImage, trainLabel, testImage, testLabel, a);
Least_Squares_Comparison = Least_Squares_Label == testLabel;
Least_Squares_Error_Rate = (1 - sum(Least_Squares_Comparison)/500) * 100;
Least_Squares_class_Error_Rates = Class_Error(Least_Squares_Label, testLabel);

%part c
Label = nearest_Neighbor(trainImage, trainLabel, testImage, testLabel);
Comparison = Label == testLabel;
Error_Rate = (1 - sum(Comparison)/500) * 100;
class_Error_Rates = Class_Error(Label, testLabel);

function Labels = least_Squares_Nearest_Neighbor(trainImage, trainLabel, testImage, testLabel, A)
    temp_Distance = [];
    Labels = [];
    for j = 1:500
        for i = 1:5000
           dist = (testImage(:, :, j) - A * trainImage(:, :, i)) .^ 2;
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