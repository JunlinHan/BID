clear all;
% The folder to your generated/separated images.
folder1 ='your_path';
files1 = dir(folder1);

image_num=1329; %choose how many images to process, 185 for rain streak, 
% 249 for rain drop, and 1329 for snow.
count_brisque=0;
count_niqe=0;  
for i=3:image_num+2
image1=uint8(imread(strcat(folder1,'\',files1(i).name)));
bri_score = brisque(image1);
niq_score = niqe(image1);
count_brisque = count_brisque + bri_score;
count_niqe = count_niqe + niq_score;
end
count_brisque=count_brisque/image_num;
count_niqe=count_niqe/image_num;
disp("BRISQUE result");
disp(count_brisque);
disp("NIQE result");
disp(count_niqe);











    

