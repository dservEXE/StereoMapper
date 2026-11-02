"""Optional metric Census/4-path SGM on NVIDIA, with no neural depth guesses.

CUDA kernels use signed disparities, independent left/right searches and subpixel
parabolic refinement. CPU consistency/texture/geometry guards remain in DepthEngine.
"""
import os
import sys
import warnings
from pathlib import Path
from functools import lru_cache
import numpy as np

SOURCE = r'''
extern "C" __global__ void census(const unsigned char* a, unsigned long long* out,int H,int W) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=H*W) return;
    int y=p/W,x=p%W; unsigned long long bits=0; int center=a[p];
    for(int j=-3;j<=3;j++) for(int i=-4;i<=4;i++) {
        if(i==0 && j==0) continue;
        int xx=max(0,min(W-1,x+i)),yy=max(0,min(H-1,y+j));
        bits=(bits<<1)|(a[yy*W+xx]<center);
    }
    out[p]=bits;
}
extern "C" __global__ void costs(const unsigned long long* a,const unsigned long long* b,
    const unsigned char* ga,const unsigned char* gb,unsigned char* out,int H,int W,int D,int minimum) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=H*W*D) return;
    int p=k/D,d=k%D+minimum,x=p%W,xr=x-d;
    if(xr>=2 && xr<W-2 && x>=2 && x<W-2 && p/W>=2 && p/W<H-2) {
        int q=p-d,l=ga[p],r=gb[q];
        int delta=min(abs(l-r),min(abs(l-(r+gb[q-1])/2),abs(l-(r+gb[q+1])/2)));
        out[k]=__popcll(a[p]^b[q])+min(16,delta/4);
    } else out[k]=127;
}
extern "C" __global__ void cost_horizontal(const unsigned char* input,unsigned short* output,int H,int W,int D) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=H*W*D) return;
    int p=k/D,d=k%D,x=p%W,y=p/W,total=0;
    for(int i=-2;i<=2;i++) {
        int xx=max(0,min(W-1,x+i));
        total+=input[(y*W+xx)*D+d];
    }
    output[k]=total;
}
extern "C" __global__ void cost_vertical(const unsigned short* input,unsigned char* output,int H,int W,int D) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=H*W*D) return;
    int p=k/D,d=k%D,x=p%W,y=p/W,total=0;
    for(int j=-2;j<=2;j++) {
        int yy=max(0,min(H-1,y+j));
        total+=input[(yy*W+x)*D+d];
    }
    output[k]=(total+12)/25;
}
extern "C" __global__ void aggregate(const unsigned char* cost,unsigned short* sum,
    const unsigned char* gray,int H,int W,int D,int axis,int direction) {
    // One block follows a scanline. Disparities are parallel threads.
    extern __shared__ int shared[];
    int* prev=shared; int* curr=shared+blockDim.x; int* reduction=shared+2*blockDim.x;
    int d=threadIdx.x,line=blockIdx.x,length=axis==0?W:H;
    prev[d]=0; __syncthreads();
    for(int t=0;t<length;t++) {
        int s=direction>0?t:length-1-t;
        int p=axis==0?line*W+s:s*W+line;
        int minimum=d<D?prev[d]:100000;
        for(int shift=16;shift>0;shift>>=1)
            minimum=min(minimum,__shfl_down_sync(0xffffffff,minimum,shift));
        if((d&31)==0) reduction[d/32]=minimum;
        __syncthreads();
        if(d<32) {
            minimum=d<blockDim.x/32?reduction[d]:100000;
            for(int shift=16;shift>0;shift>>=1)
                minimum=min(minimum,__shfl_down_sync(0xffffffff,minimum,shift));
            if(d==0) reduction[31]=minimum;
        }
        __syncthreads();
        int smallest=reduction[31];
        if(d<D) {
            int before=p-(axis==0?direction:direction*W);
            int contrast=t?abs((int)gray[p]-(int)gray[before]):0;
            int penalty=max(16,64/(1+contrast/8));
            int value=min(prev[d],smallest+penalty);
            if(d) value=min(value,prev[d-1]+8);
            if(d+1<D) value=min(value,prev[d+1]+8);
            curr[d]=(int)cost[p*D+d]+value-smallest;
            sum[p*D+d]+=curr[d];
        }
        __syncthreads();
        prev[d]=d<D?curr[d]:100000;
        __syncthreads();
    }
}
extern "C" __global__ void select_disp(const unsigned short* sum,short* out,
    int H,int W,int D,int minimum,int uniqueness) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=H*W) return;
    const unsigned short* row=sum+p*D;
    int best=0,value=100000;
    for(int d=0;d<D;d++) if(row[d]<value){value=row[d];best=d;}
    int second=100000;
    for(int d=0;d<D;d++) if(abs(d-best)>1) second=min(second,(int)row[d]);
    int x=p%W,disparity=minimum+best,xr=x-disparity;
    if(best==0 || best==D-1 || xr<2 || xr>=W-2 || x<2 || x>=W-2 ||
        second*(100-uniqueness)<=value*100) {out[p]=(minimum-1)*16;return;}
    float a=row[best-1],b=row[best],c=row[best+1];
    float offset=(a-c)/fmaxf(2*(a+c-2*b),1.f);
    out[p]=(short)roundf((disparity+fminf(.5f,fmaxf(-.5f,offset)))*16.f);
}
extern "C" __global__ void reject_ambiguous(const unsigned char* cost,short* out,
    int pixels,int D,int minimum) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=pixels || out[p]<minimum*16) return;
    int best=(int)roundf(out[p]/16.f)-minimum;
    if(best<0 || best>=D) return;
    const unsigned char* row=cost+p*D;
    // Subpixel SGM refinement can put the winner one bin off a raw-cost minimum.
    int centre=best;
    for(int d=max(0,best-1);d<=min(D-1,best+1);d++)
        if(row[d]<row[centre]) centre=d;
    if(centre<3 || centre>=D-3) return;
    int chosen=row[centre], margin=max(1,chosen/10), prominence=max(6,chosen/2);
    int cheapest=chosen;
    for(int d=0;d<D;d++) cheapest=min(cheapest,(int)row[d]);
    // A much better photometric candidate is not evidence of repetition: SGM
    // may legitimately prefer a locally weaker match supported by its context.
    if(chosen>cheapest+margin) return;
    int left=0,right=0;
    for(int offset=1;offset<=3;offset++) {
        left=max(left,(int)row[centre-offset]); right=max(right,(int)row[centre+offset]);
    }
    // A broad or flat valley is common on real surfaces. Its neighbouring bins
    // are not separate aliases, and SGM/LR/WLS already assess their reliability.
    if(min(left,right)<chosen+prominence) return;
    for(int direction=-1;direction<=1;direction+=2) {
        int barrier=chosen;
        for(int distance=1;distance<D;distance++) {
            int d=centre+direction*distance;
            if(d<0 || d>=D) break;
            barrier=max(barrier,(int)row[d]);
            if(distance<4 || d<3 || d>=D-3) continue;
            int alternative=row[d];
            if(abs(alternative-chosen)>margin || alternative>row[d-1] || alternative>row[d+1]) continue;
            int before=0,after=0;
            for(int offset=1;offset<=3;offset++) {
                before=max(before,(int)row[d-offset]); after=max(after,(int)row[d+offset]);
            }
            // Repeated keys produce two distinct, similarly deep minima with a
            // pronounced ridge between them. Do not reject an ordinary plateau.
            if(min(before,after)>=alternative+prominence &&
                barrier>=max(chosen,alternative)+2*prominence) {
                out[p]=(minimum-1)*16; return;
            }
        }
    }
}
'''


@lru_cache(maxsize=1)
def runtime():
    # Keep cache in an ASCII path on Windows (NVRTC cannot reliably use Cyrillic).
    os.environ.setdefault("CUPY_CACHE_DIR",os.path.join(os.environ.get("LOCALAPPDATA",os.getcwd()),"stereo-cuda-cache"))
    toolkit=Path(sys.prefix)/"Lib/site-packages/nvidia/cuda_runtime"
    if toolkit.exists():
        os.environ.setdefault("CUDA_PATH",str(toolkit))
        # Component wheels contain NVRTC and headers, but deliberately no nvcc.
        # CuPy's legacy toolkit probe warns even when the component runtime works.
        warnings.filterwarnings("ignore",message="CUDA path could not be detected.*",module=r"cupy\._environment")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<1:
        raise RuntimeError("NVIDIA CUDA не найдена.")
    module=cp.RawModule(code=SOURCE,options=("--std=c++11",))
    kernels={name:module.get_function(name) for name in ("census","costs","cost_horizontal","cost_vertical","aggregate","select_disp","reject_ambiguous")}
    return cp,kernels


def gpu_info():
    try:
        cp,_=runtime()
        prop=cp.cuda.runtime.getDeviceProperties(0)
        name=prop["name"]
        return name.decode() if isinstance(name,bytes) else str(name)
    except Exception:
        return None


class CudaSGM:
    def __init__(self, number, minimum=0, uniqueness=5, ambiguity_guard=False):
        self.number,self.minimum,self.uniqueness=int(number),int(minimum),int(uniqueness)
        self.ambiguity_guard=ambiguity_guard

    def compute(self, a, b):
        cp,kernels=runtime()
        h,w=a.shape; d=self.number
        if d>512:
            raise ValueError("CUDA SGM поддерживает до 512 положений в одном проходе.")
        if h*w*d>1_000_000_000:
            raise ValueError("Слишком большой объём поиска для CUDA: уменьшите ширину или диапазон.")
        # Reusable CuPy allocator caches these buffers between frames/bands.
        left,right=cp.asarray(np.ascontiguousarray(a)),cp.asarray(np.ascontiguousarray(b))
        ca,cb=cp.empty((h,w),cp.uint64),cp.empty((h,w),cp.uint64)
        pixels=h*w
        for image,out in ((left,ca),(right,cb)):
            kernels["census"](((pixels+255)//256,),(256,),(image,out,np.int32(h),np.int32(w)))
        raw_cost=cp.empty((h,w,d),cp.uint8)
        horizontal=cp.empty((h,w,d),cp.uint16)
        cost=cp.empty((h,w,d),cp.uint8)
        sums=cp.zeros((h,w,d),cp.uint16)
        kernels["costs"](((pixels*d+255)//256,),(256,),
            (ca,cb,left,right,raw_cost,np.int32(h),np.int32(w),np.int32(d),np.int32(self.minimum)))
        kernels["cost_horizontal"](((pixels*d+255)//256,),(256,),
            (raw_cost,horizontal,np.int32(h),np.int32(w),np.int32(d)))
        kernels["cost_vertical"](((pixels*d+255)//256,),(256,),
            (horizontal,cost,np.int32(h),np.int32(w),np.int32(d)))
        threads=max(32,1<<(d-1).bit_length())
        for axis in (0,1):
            for direction in (1,-1):
                kernels["aggregate"]((h if axis==0 else w,),(threads,),
                    (cost,sums,left,np.int32(h),np.int32(w),np.int32(d),np.int32(axis),np.int32(direction)),
                    shared_mem=threads*3*4)
        out=cp.empty((h,w),cp.int16)
        kernels["select_disp"](((pixels+255)//256,),(256,),
            (sums,out,np.int32(h),np.int32(w),np.int32(d),np.int32(self.minimum),np.int32(self.uniqueness)))
        if self.ambiguity_guard:
            kernels["reject_ambiguous"](((pixels+255)//256,),(256,),
                (cost,out,np.int32(pixels),np.int32(d),np.int32(self.minimum)))
        return cp.asnumpy(out)
