% addpath('mex');

use_rectified = 1;
delta = 1;
use_full_image = 0;

imdir = '~/robot/depth-estimation/data/no-risk/part7/';
lst = dir([imdir 'images']);

formatSpec = '%09.0f';
h = waitbar(0,'Please wait..');

nImg = size(lst,1)-5;

for i=1:nImg
	waitbar(i/nImg)
    if use_rectified
        im1 = im2double(imread([imdir 'rectified_images/' num2str(i,formatSpec) '.jpg']));
    else
        im1 = im2double(imread([imdir 'undistorted_images/' num2str(i,formatSpec) '.jpg']));
    end
	im2 = im2double(imread([imdir 'undistorted_images/' num2str(i+delta,formatSpec) '.jpg']));

    w = 320;
    h = size(im1, 1)*w/size(im1, 2);
    if use_full_image == 0
        im1 = imresize(im1,[h w],'nearest');
        im2 = imresize(im2,[h w],'nearest');
    end
    
	% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
	alpha = 0.02;
        ratio = 0.75;
	minWidth = 60;
	nOuterFPIterations = 7;
	nInnerFPIterations = 1;
	nSORIterations = 30;

	para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

	% this is the core part of calling the mexed dll file for computing optical flow
	% it also returns the time that is needed for two-frame estimation
	tic;
	[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
	toc;
    
    if use_full_image
        vx = imresize(vx, [h, w], 'bilinear');
        vy = imresize(vy, [h, w], 'bilinear');
        vx = double(w)/double(size(im1, 2))*vx;
        vy = double(h)/double(size(im1, 1))*vy;
        vx = round(vx);
        vy = round(vy);
    end

	% visualize flow field
	clear flow;
	flow(:,:,1) = cast(vy+128,'uint8');
	flow(:,:,2) = cast(vx+128,'uint8');
	flow(:,:,3) = 0;
	% imflow = flowToColor(flow);
	% max(max(flow))
	% min(min(flow))

	% figure; imshow(flow(:,:,1));
	% figure; imshow(flow(:,:,2));

    if use_rectified
        output = sprintf('%srectified_flow2/%dx%d/celiu/%d/', imdir, w, h, delta);
    else
        output = sprintf('%sflow/%dx%d/celiu/%d/', imdir, w, h, delta);
    end
    system(['mkdir -p ' output]);
	imwrite(flow, [output num2str(i+delta,formatSpec) '.png'], 'png');
end

close(h)

% im = imread([output num2str(i,formatSpec) '.png']);
% max(max(im))
% min(min(im))

% vy = im(:,:,1);
% vx = im(:,:,2);
% vz = im(:,:,3);


