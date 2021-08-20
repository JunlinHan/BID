%% compute rmse for ISTD (V1)
clear;close all;clc

% mask directory|mask
maskdir = 'C:\Users\76454\Desktop\BID\datasets\jointremoval_v1\testB\';
MD = dir([maskdir '/*.png']);

% ground truth directory|GT
freedir = 'C:\Users\76454\Desktop\BID\datasets\jointremoval_v1\testA\'; 
FD = dir([freedir '/*.png']);

% predicted result directory
shadowdir = 'C:\Users\76454\Desktop\BID\results\task3_v1\test_latest\images\fake_A\';
SD = dir([shadowdir '/*.png']);


total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
rl=zeros(1,size(SD,1)); 
ra=zeros(1,size(SD,1));
rb=zeros(1,size(SD,1));
nrl=zeros(1,size(SD,1));
nra=zeros(1,size(SD,1));
nrb=zeros(1,size(SD,1));
srl=zeros(1,size(SD,1));
sra=zeros(1,size(SD,1));
srb=zeros(1,size(SD,1));
ppsnr=zeros(1,size(SD,1));
ppsnrs=zeros(1,size(SD,1));
ppsnrn=zeros(1,size(SD,1));
sssim=zeros(1,size(SD,1));
sssims=zeros(1,size(SD,1));
sssimn=zeros(1,size(SD,1));
% ISTD dataset image size 480*640
tic;
mask = ones([480,640]);
cform = makecform('srgb2lab');

for i=1:size(SD)
    sname = strcat(shadowdir,SD(i).name); 
    fname = strcat(freedir,FD(i).name); 
    mname = strcat(maskdir,MD(i).name); 
    s=imread(sname);
    f=imread(fname);
    m=imread(mname);
    s=imresize(s,[256 256]);
    f=imresize(f,[256 256]);
    m=imresize(m,[256 256]);
    mask = ones([size(f,1),size(f,2)]);

    nmask=~m;       
    smask=~nmask;   
    
    f = double(f)/255;
    s = double(s)/255;
    
  
    f = applycform(f,cform);
    s = applycform(s,cform);

    
    %abs lab
    absl=abs(f(:,:,1) - s(:,:,1));
    absa=abs(f(:,:,2) - s(:,:,2));
    absb=abs(f(:,:,3) - s(:,:,3));

    % rmse
    summask=sum(mask(:));
    rl(i)=sum(absl(:))/summask;
    ra(i)=sum(absa(:))/summask;
    rb(i)=sum(absb(:))/summask;
    
    %% non-shadow, ours, per image
    distl = absl.* nmask;
    dista = absa.* nmask;
    distb = absb.* nmask;
    sumnmask=sum(nmask(:));
    nrl(i)=sum(distl(:))/sumnmask;
    nra(i)=sum(dista(:))/sumnmask;
    nrb(i)=sum(distb(:))/sumnmask;

    %% rmse in shadow, original way, per pixel
    dist = abs((f - s).* repmat(smask,[1 1 3]));
    total_dists = total_dists + sum(dist(:));
    total_pixels = total_pixels + sum(smask(:));
    % rmse in non-shadow, original way, per pixel
    dist = abs((f - s).* repmat(nmask,[1 1 3]));
    total_distn = total_distn + sum(dist(:));
    total_pixeln = total_pixeln + sum(nmask(:));  
end
toc;
%% rmse in shadow, original way, per pixel
fprintf('\tall,\tnon-shadow,\tshadow:\n%f\t%f\t%f\n\n',mean(rl)+mean(ra)+mean(rb),total_distn/total_pixeln,total_dists/total_pixels);

