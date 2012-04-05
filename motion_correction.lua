require 'torch'
require 'opencv'
require 'common'

w_center = 0
h_center = 0

function ddot(a ,b)
	p = a[1]*b[1] + a[2]*b[2]
	return p
end

function get_inliers_number(ptsin, ptsout, sample)

	local inliers = 0

	local nsamples = sample.inpts:size(1)
	local npts = ptsin:size(1)
	
	local Hs,x = lsq_trans(sample.inpts, sample.outpts, w_center, h_center)
	
	local d = torch.Tensor(npts)

	local p = {-1, 0}
	local t = {0, -1}

	local pt = {}
	local mpt = {}
	
	for i=1,npts do
		pt = {ptsin[i][1] - w_center, ptsin[i][2] - h_center}
		mpt = {ptsout[i][1] - w_center, ptsout[i][2] - h_center}

		local f = {mpt[1] - pt[1], mpt[2] - pt[2]}
		local z = {-pt[1], -pt[2]}
		local r = {pt[2], -pt[1]}

		local fout = {x[1]*p[1] + x[2]*t[1] + x[3]*z[1] + x[4]*r[1],
				x[1]*p[2] + x[2]*t[2] + x[3]*z[2] + x[4]*r[2]}
		
		local err = {fout[1] - f[1], fout[2] - f[2]}

		d[i] = torch.sqrt(torch.pow(err[1], 2) + torch.pow(err[2], 2))
	end

	local stderr = torch.std(d)
	local thr = torch.sqrt(5.99*torch.pow(stderr,2))
	
	for i=1,npts do
		if d[i]<=thr then inliers = inliers+1 end
	end

	return inliers
end

function get_random_sample(ptsin, ptsout, s)
	local sample = {}
	sample.inpts = torch.Tensor(s, 2)
	sample.outpts = torch.Tensor(s, 2)
	local npts = ptsin:size(1)
	for i=1,s do
		local idx = randInt(1,npts+1)
		sample.inpts[i] = ptsin[idx]
		sample.outpts[i] = ptsout[idx]
	end

	return sample
end

function calculate_samples_number(ptsin, ptsout, s)
	local npts = ptsin:size(1)
	local p = 0.99
	local N = 1e12
	local sample_count = 0
	local inliers = 1
	local err
	while (N > sample_count) do
		sample = get_random_sample(ptsin, ptsout, s)
		inliers = math.max(get_inliers_number(ptsin, ptsout, sample), 1)
		err = 1 - inliers/npts
		N = math.log(1-p)/math.log(1-math.pow(1-err, s))
		sample_count = sample_count+1
	end
	
	return math.max(math.ceil(N),1)
end


function lsq_trans(ptsin, ptsout, w_center, h_center)

	local A = torch.Tensor(4, 4):fill(0)
	local b = torch.Tensor(4, 1):fill(0)
	local H = torch.Tensor(2, 3)

	local p = {-1, 0}
	local t = {0, -1}

	local pt = {}
	local mpt = {}

	for i=1,ptsin:size(1) do
		pt = {ptsin[i][1] - w_center, ptsin[i][2] - h_center}
		mpt = {ptsout[i][1] - w_center, ptsout[i][2] - h_center}

		local f = {mpt[1] - pt[1], mpt[2] - pt[2]}
		local z = {-pt[1], -pt[2]}
		local r = {pt[2], -pt[1]}

		A[1][1] = A[1][1] + ddot(p,p)
		A[1][2] = A[1][2] + ddot(p,t)
		A[1][3] = A[1][3] + ddot(p,z)
		A[1][4] = A[1][4] + ddot(p,r)
		A[2][1] = A[2][1] + ddot(t,p)
		A[2][2] = A[2][2] + ddot(t,t)
		A[2][3] = A[2][3] + ddot(t,z)
		A[2][4] = A[2][4] + ddot(t,r)
		A[3][1] = A[3][1] + ddot(z,p)
		A[3][2] = A[3][2] + ddot(z,t)
		A[3][3] = A[3][3] + ddot(z,z)
		A[3][4] = A[3][4] + ddot(z,r)
		A[4][1] = A[4][1] + ddot(r,p)
		A[4][2] = A[4][2] + ddot(r,t)
		A[4][3] = A[4][3] + ddot(r,z)
		A[4][4] = A[4][4] + ddot(r,r)

		b[1][1] = b[1][1] + ddot(p,f)
		b[2][1] = b[2][1] + ddot(t,f)
		b[3][1] = b[3][1] + ddot(z,f)	
		b[4][1] = b[4][1] + ddot(r,f)
	end

	x = torch.gesv(b,A)
	
	local dtheta = -torch.atan(x[4][1])
	local dx = x[1][1]
	local dy = x[2][1]

	H[1][1] = torch.cos(dtheta)
	H[1][2] = torch.sin(dtheta)
	H[1][3] = ((1-torch.cos(dtheta))*w_center - torch.sin(dtheta)*h_center) + dx
	H[2][1] = -torch.sin(dtheta)
	H[2][2] = torch.cos(dtheta)
	H[2][3] = (torch.sin(dtheta)*w_center - (1-torch.cos(dtheta))*h_center) + dy

	return H, x
end

function lsq_trans_ransac(ptsin, ptsout, xcenter, ycenter)

	w_center = xcenter
	h_center = ycenter

	-- sample size
	local s = 5

	local N = calculate_samples_number(ptsin, ptsout, s)
	local Hs = torch.Tensor(N, 2, 3)
	local inliers = torch.Tensor(N)

	for i=1,N do
		local sample = get_random_sample(ptsin, ptsout, s)
		Hs[i] = lsq_trans(sample.inpts, sample.outpts, w_center, h_center)
		inliers[i] = get_inliers_number(ptsin, ptsout, sample)
	end

	maxinliers,idx = torch.max(inliers,1)
	return Hs[idx[1]],maxinliers[1]
end

function test_lsq_trans()
	imgfilenameL = 'data/parc/images/000000000.jpg'
	imgfilenameR = 'data/parc/images/000000020.jpg'
	imgL = image.loadJPG(imgfilenameL)
	imgR = image.loadJPG(imgfilenameR)

	imgL = image.scale(imgL, 320, 240)
	imgR = image.scale(imgR, 320, 240)

	w_imgs = imgL:size(3)
	h_imgs = imgL:size(2)
	local w_center = w_imgs/2
	local h_center = h_imgs/2

	local ptsin = opencv.GoodFeaturesToTrack{image=imgL, count=50}
	local ptsout = opencv.TrackPyrLK{pair={imgL,imgR}, points_in=ptsin}

	opencv.drawFlowlinesOnImage({ptsin,ptsout},imgL)

	local H = lsq_trans(ptsin, ptsout, w_center, h_center)
	print('H using all data:')
	print('(using ' .. ptsin:size(1) .. ' points)')
	print(H)

	local warpimg = opencv.WarpAffine(imgR, H)
	local ptsoutw = opencv.TrackPyrLK{pair={imgL,warpimg},points_in=ptsin}
	opencv.drawFlowlinesOnImage({ptsin,ptsoutw},warpimg)
	image.display{image={imgL,warpimg},legend='Original'}

	local Hr,m = lsq_trans_ransac(ptsin, ptsout, w_center, h_center)
	print('H using RANSAC:')
	print('(using ' .. m .. ' inliers)')
	print(Hr)

	local warpimgrsac = opencv.WarpAffine(imgR, Hr)
	local ptsoutwrsac = opencv.TrackPyrLK{pair={imgL,warpimgrsac},points_in=ptsin}
	opencv.drawFlowlinesOnImage({ptsin,ptsoutwrsac},warpimgrsac)
	image.display{image={imgL,warpimgrsac},legend='RANSAC'}

end