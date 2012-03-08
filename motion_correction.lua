require 'torch'
require 'opencv'

function ddot(a ,b)
	p = a[1]*b[1] + a[2]*b[2]
	return p
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

		f = {mpt[1] - pt[1], mpt[2] - pt[2]}
		z = {-pt[1], -pt[2]}
		r = {pt[2], -pt[1]}

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

	H[1][1] = torch.cos(dtheta)
	H[1][2] = torch.sin(dtheta)
	H[1][3] = ((1-torch.cos(dtheta))*w_center - torch.sin(dtheta)*h_center) + x[1][1]
	H[2][1] = -torch.sin(dtheta)
	H[2][2] = torch.cos(dtheta)
	H[2][3] = (torch.sin(dtheta)*w_center - (1-torch.cos(dtheta))*h_center) + x[2][1]

	return H
end
