thresh=0.5;
for i=1:200
name=sprintf('input_%03d_input.jpg',i);
I=imread(name);
I0=im2bw(I,thresh);
I1=im2uint8(I0);
j=find('.'==name);
imname=name(1:j-1);
imwrite(I1,strcat(imname,'.bmp'));
end