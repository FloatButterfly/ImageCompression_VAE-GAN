k = 0;
for q=0:199
name=sprintf('image_copy/input_%03d_input.png',q);
I=imread(name);
height= size(I, 1); %�����
width = size(I, 2); %�����
region_size = 64;  %�����ߴ�С
numRow = round(height/region_size);%ͼ���ڴ�ֱ�����ֳܷɶ��ٸ���СΪregion_size
numCol = round(width/region_size);%ͼ����ˮƽ�����ֳܷɶ��ٸ���СΪregion_size
I=imresize(I,[numRow*region_size,numCol*region_size]);%���������µ�ͼ���Է�ֹtemp�±�Խ��
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
