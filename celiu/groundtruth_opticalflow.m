% addpath('mex');

imdir = '~/data/sfm/indoor/';
lst = dir([imdir 'images']);

formatSpec = '%09.0f';
h = waitbar(0,'Please wait..');

nImg = size(lst,1)-5;

for i=0:nImg
	waitbar(i/nImg)
	im1 = im2double(imread([imdir 'images/' num2str(i,formatSpec) '.png']));
	im2 = im2double(imread([imdir 'images/' num2str(i+1,formatSpec) '.png']));

	im1 = imresize(im1,[180 320],'bicubic');
	im2 = imresize(im2,[180 320],'bicubic');

	% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
	alpha = 0.012;
	ratio = 0.75;
	minWidth = 20;
	nOuterFPIterations = 7;
	nInnerFPIterations = 1;
	nSORIterations = 30;

	para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

	% this is the core part of calling the mexed dll file for computing optical flow
	% it also returns the time that is needed for two-frame estimation
	tic;
	[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
	toc;

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

	output = [imdir 'flow/320x180/celiu/'];
	imwrite(flow, [output num2str(i+1,formatSpec) '.png'], 'png');
end

close(h)

% im = imread([output num2str(i,formatSpec) '.png']);
% max(max(im))
% min(min(im))

% vy = im(:,:,1);
% vx = im(:,:,2);
% vz = im(:,:,3);


