clear all;
% The folder to your generated/separated images.
folder1 ='C:\Users\76454\Desktop\BID\results\biden3\test_latest\images\fake_B';
% The folder to real images.
folder2= 'C:\Users\76454\Desktop\BID\results\biden3\test_latest\images\real_B';
files1 = dir(folder1);
files2 = dir(folder2);

image_num=300; %choose how many images to process, 300 for task I, 500 for task II.
count_ssim=0;
count_psnr=0;
for i=3:image_num+2
image1=uint8(imread(strcat(folder1,'\',files1(i).name)));
image2=uint8(imread(strcat(folder2,'\',files2(i).name)));
[ssimval,ssimmap]=ssim(image1,image2);
[peaksnr, snr] = psnr(image1,image2);
count_ssim = count_ssim + ssimval;
count_psnr = count_psnr + peaksnr;
end
count_ssim=count_ssim/image_num;
count_psnr=count_psnr/image_num;
disp("SSIM result");
disp(count_ssim);
disp("PSNR result");
disp(count_psnr);











    

