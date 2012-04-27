#define N_MAX_RATIOS 10
#include<math.h>
#define mod(a, b) ((a) %% (b))

// get args
const void* idlong = luaT_checktypename2id(L, "torch.LongTensor");
THLongTensor *xIm = luaT_checkudata(L, 1, idlong);
int maxh = lua_tonumber(L, 2);
int maxw = lua_tonumber(L, 3);
int nratios = luaL_getn(L, 4);
//int* ratios = (int*)malloc(sizeof(int)*nratios);
//assert(nratios < N_MAX_RATIOS);
int ratios[N_MAX_RATIOS];
int i, j, k;
for (i = 0; i < nratios; ++i) {
  lua_pushnumber(L, i);
  lua_gettable(L, 4);
  ratios[i] = lua_tonumber(L, -1);
}
THLongTensor *retx = luaT_checkudata(L, 5, idlong);
THLongTensor *rety = luaT_checkudata(L, 6, idlong);

// get useful variables
int chmaxh = ceil(maxh/2);
int chmaxw = ceil(maxw/2);
int height = xIm->size[0];
int width = xIm->size[1];
long *xIms = xIm->stride;
long *retxs = retx->stride;
long *retys = rety->stride;
long* xIm_p = THLongTensor_data(xIm);
long* retx_p = THLongTensor_data(retx);
long* rety_p = THLongTensor_data(rety);
int patcharea = maxh*maxw;
//int* borders = (int*)malloc(sizeof(int)*nratios);
//int* lengths = (int*)malloc(sizeof(int)*nratios);
int borders[N_MAX_RATIOS];
int lengths[N_MAX_RATIOS];
for (i = 1; i < nratios; ++i) {
  borders[i] = round((float)maxw*((float)ratios[i]-(float)ratios[i-1])/(2.0f*(float)ratios[i]));
  lengths[i] = 2 * maxw + 2 * (maxh - 2 * borders[i]) * borders[i];
}
long x;
int d, mH;

// loop over pixels;
for (i = 0; i < height; ++i) {
  for (j = 0; j < width; ++j) {
    x = xIm_p[xIms[0]*i + xIms[1]*j];
    if (x < patcharea) {
      // higher resolution : full patch used
      rety_p[retys[0]*i + retys[1]*j] = floor((x-1)/maxw) + 1 - chmaxh;
      retx_p[retxs[0]*i + retxs[1]*j] = mod(x-1, maxw) + 1 - chmaxw;
    } else {
      // smaller resolution : middle area isn't used
      x -= patcharea;
      for (k = 1; k < nratios; ++k) {
	d = borders[k];
	mH = (maxh-2*d)*d;
	if (x <= lengths[k]) {
	  if (x < d*maxw) {
	    rety_p[retys[0]*i + retys[1]*j] = (floor((x-1)/maxw) + 1 - chmaxh) * ratios[k];
	    retx_p[retxs[0]*i + retxs[1]*j] = (mod(x-1, maxw) + 1 - chmaxw) * ratios[k];
	    break;
	  }
	  x -= d*maxw;
	  if (x <= mH) {
	    rety_p[retys[0]*i + retys[1]*j] = (floor((x-1)/d) + 1 + d - chmaxh) * ratios[k];
	    retx_p[retxs[0]*i + retxs[1]*j] = (mod(x-1, d) + 1 - chmaxw) * ratios[k];
	    break;
	  }
	  x -= mH;
	  if (x <= mH) {
	    rety_p[retys[0]*i + retys[1]*j] = (floor((x-1)/d) + 1 + d - chmaxh) * ratios[k];
	    retx_p[retxs[0]*i + retxs[1]*j] = (mod(x-1, d) + 1 + maxw-d - chmaxw) * ratios[k];
	    break;
	  }
	  x -= mH;
	  if (x < d*maxw) {
	    rety_p[retys[0]*i + retys[1]*j] = (floor((x-1)/maxw)+1+maxh-d - chmaxh) * ratios[k];
	    retx_p[retxs[0]*i + retxs[1]*j] = (mod(x-1, maxw) + 1 - chmaxw) * ratios[k];
	    break;
	  }
	  //assert(0); // this should not happen if the code is correct
	} else {
	  x -= lengths[k];
	}
      }
      //assert(k < nratios); // this should not happen if geometry is coherent with x
    }
  }
}
//free(ratios);
//free(borders);
//free(lengths);
