load("dataset-letters.mat");
images = getfield(dataset,"images");
labels = getfield(dataset,"labels");
truelabels = categorical(labels);
key = getfield(dataset,"key");
keys = num2cell(key);
letterLabel = keys(labels);
trueletterLabels = categorical(letterLabel);
newLLab = transpose(trueletterLabels);
figure(1);


[trainSet,valSet,testSet] = dividerand(26000,50,0,50);
trainImages = images(trainSet,:);
testImages = images(testSet,:);
trainLabels = newLLab(trainSet);
testLabels = newLLab(testSet);
testPredictions = categorical.empty(size(testImages,1),0);
testPredictions2 = categorical.empty(size(testImages,1),0);

for i = 1:12
    reshapedIm = trainImages(i,:);
    realIm = reshape(reshapedIm,[28,28]);
    subplot(3,4,i);
    imshow(realIm);
    title(trainLabels(i));
end

saveas(gcf,"CW1ImageSet.png");

tic;
for i = 1:size(testImages,1)
    comp1 = trainImages;
    comp2 = repmat(testImages(i,:),[size(trainImages,1),1]);
    distance1 = sum((comp1-comp2).^2,2);%L2
    [~,ind] = sort(distance1);
    ind = ind(1:5);
    labs = trainLabels(ind);
    testPredictions(i,1) = mode(labs);
end

figure(2);
correct_Predictions = sum(testLabels==testPredictions);
accuracy = correct_Predictions/size(testLabels,1);
confusionchart(testLabels,testPredictions);
toc;
timerEnd = toc;

tic;
for i = 1:size(testImages,1)
    comp1 = trainImages;
    comp2 = repmat(testImages(i,:),[size(trainImages,1),1]);
    distance2 = sum(abs(comp1-comp2),2);%L1
    [~,ind2] = sort(distance2);
    ind2 = ind2(1:5);
    labs2 = trainLabels(ind2);
    testPredictions2(i,1) = mode(labs2);
end

figure(5)
correct_Predictions4 = sum(testLabels == testPredictions2);
accuracy4 = correct_Predictions4/size(testLabels,1);
confusionchart(testLabels,testPredictions2);
toc;
timerEnd4 = toc;

figure(3);
tic;
dc = fitcknn(trainImages,trainLabels);
predicted = predict(dc,testImages);
correct_Predictions2 = sum(testLabels == predicted);
accuracy2 = correct_Predictions2 /size(testLabels,1);
knnmodelCM2 = confusionchart(testLabels,predicted);
toc;
timerEnd2 = toc;

figure(4);
tic;
ac = fitctree(trainImages,trainLabels);
predicted2 = predict(ac,testImages);
correct_Predictions3 = sum(testLabels==predicted2);
accuracy3 = correct_Predictions3/size(testLabels,1);
knnmodelCM3 = confusionchart(testLabels,predicted2);
toc;
timerEnd3 = toc;







