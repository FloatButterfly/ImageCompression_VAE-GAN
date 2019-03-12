k = 0;
for q=0:199
name=sprintf('image_copy/input_%03d_input.png',q);
I=imread(name);
height= size(I, 1); %求出行
width = size(I, 2); %求出列
region_size = 64;  %区域宽高大小
numRow = round(height/region_size);%图像在垂直方向能分成多少个大小为region_size
numCol = round(width/region_size);%图像在水平方向能分成多少个大小为region_size
I=imresize(I,[numRow*region_size,numCol*region_size]);%重新生成新的图像，以防止temp下标越界
t1 = (0:numRow-1)*region_size + 1; t2 = (1:numRow)*region_size;
t3 = (0:numCol-1)*region_size + 1; t4 = (1:numCol)*region_size;
% figure; 
for i = 1 : numRow
    for j = 1 : numCol
        temp = I(t1(i):t2(i), t3(j):t4(j), :);
        newname=sprintf('edge_block/edge_%d.jpg',k);
        imwrite(temp,newname);
        k = k + 1;
%         subplot(numRow, numCol, k);
%         imshow(temp);       
    end
end
end
