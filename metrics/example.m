
load modelparameters.mat
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;
folder1 ='C:\Users\76454\OneDrive\桌面\task2ba\test_latest\images\real_input';
files1 = dir(folder1);

image_num=185; %choose how many images to process, 185/249/1329
count_niqe=0;
count_bri=0;

for i=3:image_num+2
im=uint8(imread(strcat(folder1,'\',files1(i).name)));
 niqe = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
     mu_prisparam,cov_prisparam);
bri = brisque(im);
count_niqe = count_niqe + niqe;
count_bri = count_bri + bri;
end
count_niqe=count_niqe/image_num;
count_bri=count_bri/image_num;
disp("NIQE result");
disp(count_niqe);
disp("BRISQUE result");
disp(count_bri);