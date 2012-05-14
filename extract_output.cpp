#include<cmath>
#include<iostream>
#include<cassert>
extern "C" {
#include<luaT.h>
#include<TH/TH.h>
}
using namespace std;

// sorting network-based sort
template<typename T> inline void sortswap(T* a, T* b, long incr) {
  if (*b > *a) {
    T tmp = *b;
    *b = *a;
    *a = tmp;
    tmp = *(b+incr);
    *(b+incr) = *(a+incr);
    *(a+incr) = tmp;
  }
}
template<typename T> inline void sort4(T* a, long incr) {
  sortswap(a  , a+2, incr);
  sortswap(a+1, a+3, incr);
  sortswap(a  , a+1, incr);
  sortswap(a+2, a+3, incr);
  sortswap(a+1, a+2, incr);
}
    

static int ExtractOutput(lua_State *L) {
  const void* iddouble = luaT_checktypename2id(L, "torch.DoubleTensor");
  const void* idlong   = luaT_checktypename2id(L, "torch.LongTensor"  );
  THDoubleTensor* input         = (THDoubleTensor*)luaT_checkudata(L, 1, iddouble);
  double          threshold     = lua_tonumber   (L, 2);
  double          threshold_acc = lua_tonumber   (L, 3);
  THLongTensor*   ret           = (THLongTensor*)luaT_checkudata(L, 4, idlong  );
  THLongTensor*   retgd         = (THLongTensor*)luaT_checkudata(L, 5, idlong  );

  //THLongTensor_zero(ret);
  THLongTensor_zero(retgd);

  input = THDoubleTensor_newContiguous(input);
  double* input_p = THDoubleTensor_data(input);
  long*   ret_p   = THLongTensor_data  (ret  );
  long*   retgd_p = THLongTensor_data  (retgd);

  int nvalues = input->size[2];
  int h = input->size[0];
  int w = input->size[1];
  long* is  = input->stride;
  long* rs  = ret->stride;
  long* rgs = retgd->stride;

  const int maxhighs = 4;
  THDoubleTensor* highs  = THDoubleTensor_newWithSize4d(h, w, 2, maxhighs);
  assert(THDoubleTensor_isContiguous(highs));
  double* highs_p  = THDoubleTensor_data(highs );
  long*  hs =  highs->stride;

  THDoubleTensor_zero(highs);

  int i, j, k, n;
  double *highs_pe, *highs_pe_end, *input_pe, *input_pe_begin, *input_pe_end;
  input_pe = input_p;
  for (i = 0; i < h; ++i) {
    for (j = 0; j < w; ++j) {
      input_pe_begin = input_pe;
      input_pe_end = input_pe + nvalues;
      n = 0;
      for (; input_pe != input_pe_end; ++input_pe) {
	if (*input_pe > threshold) {
	  highs_pe = highs_p + hs[0]*i + hs[1]*j + n;
	  *highs_pe = *input_pe;
	  *(highs_pe + hs[2]) = (int)(input_pe - input_pe_begin)+1;
	  //cout << *(highs_pe + hs[2]) << endl;
	  ++n;
	  if (n == maxhighs) { //TODO this might be removed
	    input_pe = input_pe_end;
	    break;
	  }
	}
      }
    }
  }

  double acc;
  //double vmax;
  //int imax;
  for (i = 0; i < h; ++i) {
    for (j = 0; j < w; ++j) {
      highs_pe = highs_p + hs[0]*i + hs[1]*j;
      if (*highs_pe > 0) {
	sort4<double>(highs_pe, hs[2]);
	ret_p  [rs[0]*i  + rs[1]*j ] = *(highs_pe+hs[2]);
	for(k = 1; k < maxhighs; ++k)
	  highs_pe[k] += highs_pe[k-1];
	acc = 0;
	for (k = 0; k < maxhighs; ++k)
	  acc += highs_pe[k];
	if (acc >= threshold_acc)
	  retgd_p[rgs[0]*i + rgs[1]*j] = 1;
	/*
	vmax = *highs_pe;
	imax = *(highs_pe + hs[2]);
	highs_pe_end = highs_pe + 4;
	for (++highs_pe; highs_pe < highs_pe_end; ++highs_pe) {
	  if (vmax < *highs_pe) {
	    vmax = *highs_pe;
	    imax = *(highs_pe + hs[2]);
	  }
	}
	ret_p  [rs[0]*i + rs[1]*j] = imax;
	retgd_p[rgs[0]*i + rgs[1]*j] = 1;
	*/
      }
    }
  }
  
  THDoubleTensor_free( input);
  THDoubleTensor_free( highs);

  return 0;
}

static const struct luaL_reg extractoutput[] = {
  {"extractOutput", ExtractOutput},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_extractoutput (lua_State *L) {
  luaL_openlib(L, "extractoutput", extractoutput, 0);
  return 1;
}
