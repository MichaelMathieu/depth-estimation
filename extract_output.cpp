#include<cmath>
#include<iostream>
#include<cassert>
extern "C" {
#include<luaT.h>
#include<TH/TH.h>
}
using namespace std;

typedef THFloatTensor Tensor;
#define ID_TENSOR_STRING "torch.FloatTensor"
#define Tensor_(a) THFloatTensor_##a
typedef float real;
typedef double accreal;

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

template<typename T> inline void sort8(T* a, long incr) {
  sortswap(a  , a+1, incr);
  sortswap(a+2, a+3, incr);
  sortswap(a+4, a+5, incr);
  sortswap(a+6, a+7, incr);
  
  sortswap(a  , a+2, incr);
  sortswap(a+1, a+3, incr);
  sortswap(a+4, a+6, incr);
  sortswap(a+5, a+7, incr);
  
  sortswap(a+1, a+2, incr);
  sortswap(a+5, a+6, incr);
  sortswap(a  , a+4, incr);
  sortswap(a+3, a+7, incr);
  
  sortswap(a+1, a+5, incr);
  sortswap(a+2, a+6, incr);

  sortswap(a+1, a+4, incr);
  sortswap(a+3, a+6, incr);

  sortswap(a+2, a+4, incr);
  sortswap(a+3, a+5, incr);

  sortswap(a+3, a+4, incr);
}

static int ExtractOutput(lua_State *L) {
  const void* iddouble = luaT_checktypename2id(L, ID_TENSOR_STRING);
  const void* idlong   = luaT_checktypename2id(L, "torch.LongTensor"  );
  Tensor*     input         = (Tensor*)luaT_checkudata(L, 1, iddouble);
  Tensor*     scores        = (Tensor*)luaT_checkudata(L, 2, iddouble);
  double          threshold     = lua_tonumber   (L, 3);
  THLongTensor*   ret           = (THLongTensor*)luaT_checkudata(L, 4, idlong);

  input = Tensor_(newContiguous)(input);
  real* input_p = Tensor_(data)(input);
  long* ret_p   = THLongTensor_data(ret);
  real* scores_p = Tensor_(data)(scores);
  
  int nvalues = input->size[2];
  int h = input->size[0];
  int w = input->size[1];
  long* is  = input->stride;
  long* rs  = ret->stride;
  long* ss = scores->stride;
  
  int maxhighs = 4;
  if (threshold < 0.2)
    maxhighs = 8;
  Tensor* highs  = Tensor_(newWithSize4d)(h, w, 2, maxhighs);
  assert(Tensor_(isContiguous)(highs));
  real* highs_p  = Tensor_(data)(highs );
  long* hs =  highs->stride;

  Tensor_(zero)(highs);

  int i, j, k, n;
  real *highs_pe, *highs_pe_end, *input_pe, *input_pe_begin, *input_pe_end;
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
	  ++n;
	  if (n == maxhighs) { //TODO this might be removed
	    input_pe = input_pe_end;
	    break;
	  }
	}
      }
    }
  }

  accreal acc;
  if (maxhighs == 4) {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
	highs_pe = highs_p + hs[0]*i + hs[1]*j;
	if (*highs_pe > 0) {
	  sort4<real>(highs_pe, hs[2]);
	  ret_p  [rs[0]*i  + rs[1]*j ] = *(highs_pe+hs[2]);
	  for(k = 1; k < maxhighs; ++k)
	    highs_pe[k] += highs_pe[k-1];
	  acc = 0;
	  for (k = 0; k < maxhighs; ++k)
	    acc += highs_pe[k];
	  scores_p[ss[0]*i + ss[1]*j] = acc;
	}
      }
    }
  } else {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
	highs_pe = highs_p + hs[0]*i + hs[1]*j;
	if (*highs_pe > 0) {
	  sort8<real>(highs_pe, hs[2]);
	  ret_p  [rs[0]*i  + rs[1]*j ] = *(highs_pe+hs[2]);
	  for(k = 1; k < maxhighs; ++k)
	    highs_pe[k] += highs_pe[k-1];
	  acc = 0;
	  for (k = 0; k < maxhighs; ++k)
	    acc += highs_pe[k];
	  scores_p[ss[0]*i + ss[1]*j] = acc;
	}
      }
    }
  }
  
  Tensor_(free)( input);
  Tensor_(free)( highs);

  return 0;
}

static int ExtractOutputMarginalized(lua_State *L) {
  const void* iddouble = luaT_checktypename2id(L, ID_TENSOR_STRING);
  const void* idlong   = luaT_checktypename2id(L, "torch.LongTensor"  );
  Tensor*     input         = (Tensor*)luaT_checkudata(L, 1, iddouble);
  double          threshold     = lua_tonumber   (L, 2);
  double          threshold_acc = lua_tonumber   (L, 3);
  THLongTensor*   ret           = (THLongTensor*)luaT_checkudata(L, 4, idlong  );
  THLongTensor*   retgd         = (THLongTensor*)luaT_checkudata(L, 5, idlong  );

  //THLongTensor_zero(ret);
  THLongTensor_zero(retgd);

  input = Tensor_(newContiguous)(input);
  real* input_p = Tensor_(data)(input);
  long* ret_p   = THLongTensor_data  (ret  );
  long* retgd_p = THLongTensor_data  (retgd);
  
  int nvalues = input->size[2];
  int h = input->size[0];
  int w = input->size[1];
  long* is  = input->stride;
  long* rs  = ret->stride;
  long* rgs = retgd->stride;
  
  int maxhighs = 4;
  if (threshold < 0.2)
    maxhighs = 8;
  Tensor* highs  = Tensor_(newWithSize4d)(h, w, 2, maxhighs);
  assert(Tensor_(isContiguous)(highs));
  real* highs_p  = Tensor_(data)(highs );
  long* hs =  highs->stride;

  Tensor_(zero)(highs);

  int i, j, k, n;
  real *highs_pe, *highs_pe_end, *input_pe, *input_pe_begin, *input_pe_end;
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
	  ++n;
	  if (n == maxhighs) { //TODO this might be removed
	    input_pe = input_pe_end;
	    break;
	  }
	}
      }
    }
  }

  accreal acc;
  if (maxhighs == 4) {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
	highs_pe = highs_p + hs[0]*i + hs[1]*j;
	if (*highs_pe > 0) {
	  sort4<real>(highs_pe, hs[2]);
	  ret_p  [rs[0]*i  + rs[1]*j ] = *(highs_pe+hs[2]);
	  for(k = 1; k < maxhighs; ++k)
	    highs_pe[k] += highs_pe[k-1];
	  acc = 0;
	  for (k = 0; k < maxhighs; ++k)
	    acc += highs_pe[k];
	  if (acc >= threshold_acc)
	    retgd_p[rgs[0]*i + rgs[1]*j] = 1;
	}
      }
    }
  } else {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
	highs_pe = highs_p + hs[0]*i + hs[1]*j;
	if (*highs_pe > 0) {
	  sort8<real>(highs_pe, hs[2]);
	  ret_p  [rs[0]*i  + rs[1]*j ] = *(highs_pe+hs[2]);
	  for(k = 1; k < maxhighs; ++k)
	    highs_pe[k] += highs_pe[k-1];
	  acc = 0;
	  for (k = 0; k < maxhighs; ++k)
	    acc += highs_pe[k];
	  if (acc >= threshold_acc)
	    retgd_p[rgs[0]*i + rgs[1]*j] = 1;
	}
      }
    }
  }
  
  Tensor_(free)( input);
  Tensor_(free)( highs);

  return 0;
}

static int ExtractOutputMarginalized(lua_State *L) {
  const void* iddouble = luaT_checktypename2id(L, ID_TENSOR_STRING);
  const void* idlong   = luaT_checktypename2id(L, "torch.LongTensor"  );
  Tensor*     input         = (Tensor*)luaT_checkudata(L, 1, iddouble);
  double          threshold     = lua_tonumber   (L, 2);
  double          threshold_acc = lua_tonumber   (L, 3);
  THLongTensor*   ret           = (THLongTensor*)luaT_checkudata(L, 4, idlong  );
  THLongTensor*   retgd         = (THLongTensor*)luaT_checkudata(L, 5, idlong  );

  //THLongTensor_zero(ret);
  THLongTensor_zero(retgd);

  input = Tensor_(newContiguous)(input);
  real* input_p = Tensor_(data)(input);
  long* ret_p   = THLongTensor_data  (ret  );
  long* retgd_p = THLongTensor_data  (retgd);
  
  int nvalues = input->size[2];
  int h = input->size[0];
  int w = input->size[1];
  long* is  = input->stride;
  long* rs  = ret->stride;
  long* rgs = retgd->stride;
  
  int maxhighs = 4;
  if (threshold < 0.2)
    maxhighs = 8;
  Tensor* highs  = Tensor_(newWithSize4d)(h, w, 2, maxhighs);
  assert(Tensor_(isContiguous)(highs));
  real* highs_p  = Tensor_(data)(highs );
  long* hs =  highs->stride;

  Tensor_(zero)(highs);

  int i, j, k, n;
  real *highs_pe, *highs_pe_end, *input_pe, *input_pe_begin, *input_pe_end;
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
	  ++n;
	  if (n == maxhighs) { //TODO this might be removed
	    input_pe = input_pe_end;
	    break;
	  }
	}
      }
    }
  }

  accreal acc;
  if (maxhighs == 4) {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
	highs_pe = highs_p + hs[0]*i + hs[1]*j;
	if (*highs_pe > 0) {
	  sort4<real>(highs_pe, hs[2]);
	  ret_p  [rs[0]*i  + rs[1]*j ] = *(highs_pe+hs[2]);
	  for(k = 1; k < maxhighs; ++k)
	    highs_pe[k] += highs_pe[k-1];
	  acc = 0;
	  for (k = 0; k < maxhighs; ++k)
	    acc += highs_pe[k];
	  if (acc >= threshold_acc)
	    retgd_p[rgs[0]*i + rgs[1]*j] = 1;
	}
      }
    }
  } else {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
	highs_pe = highs_p + hs[0]*i + hs[1]*j;
	if (*highs_pe > 0) {
	  sort8<real>(highs_pe, hs[2]);
	  ret_p  [rs[0]*i  + rs[1]*j ] = *(highs_pe+hs[2]);
	  for(k = 1; k < maxhighs; ++k)
	    highs_pe[k] += highs_pe[k-1];
	  acc = 0;
	  for (k = 0; k < maxhighs; ++k)
	    acc += highs_pe[k];
	  if (acc >= threshold_acc)
	    retgd_p[rgs[0]*i + rgs[1]*j] = 1;
	}
      }
    }
  }
  
  Tensor_(free)( input);
  Tensor_(free)( highs);

  return 0;
}

static const struct luaL_reg extractoutput[] = {
  {"extractOutput", ExtractOutput},
  {"extractOutputMarginalized", ExtractOutputMarginalized},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_extractoutput (lua_State *L) {
  luaL_openlib(L, "extractoutput", extractoutput, 0);
  return 1;
}
