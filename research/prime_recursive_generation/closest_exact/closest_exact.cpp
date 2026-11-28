#include <bits/stdc++.h>
#include <omp.h>
using namespace std; using U=uint64_t; using I=__int128_t;
string fmt(I x){if(!x)return "0";string s;while(x){s+=char('0'+x%10);x/=10;}reverse(s.begin(),s.end());return s;}
struct Stat{I md=0,dd=0; U count=0,miss=0,maxd=0,maxm=0;};
int main(int argc,char**argv){
 U N=argc>1?stoull(argv[1]):100000000, M=N/2, cap=100000;
 U maxprime=M+cap+128, K=maxprime/6+1, W=(K+63)/64, pad=cap/6/64+4;
 vector<U> a(W+2*pad),b(W+2*pad); U *A=a.data()+pad,*B=b.data()+pad;
 int root=sqrt((long double)maxprime)+1;vector<bool> sp(root+1,true);sp[0]=sp[1]=false;vector<int> small;
 for(int p=2;p<=root;p++)if(sp[p]){if(p>=5)small.push_back(p);if(1LL*p*p<=root)for(int j=p*p;j<=root;j+=p)sp[j]=false;}
 double begin=omp_get_wtime();
 #pragma omp parallel for schedule(dynamic,1)
 for(U w0=0;w0<W;w0+=16384){U w1=min(W,w0+16384),lo=w0*64,hi=w1*64;fill(A+w0,A+w1,~U(0));fill(B+w0,B+w1,~U(0));
  for(U p:small){for(U z:{1ULL,5ULL}){U res=p*z%6;U* v=res==1?A:B;U low=max(p*p,6*lo+res);U k=(low+p-1)/p;k+=(z+6-k%6)%6;U j=p*k/6;for(;j<hi;j+=p)v[j/64]&=~(U(1)<<(j%64));}}
 }
 A[0]&=~U(1);
 auto isp=[&](U n){if(n==2||n==3)return true;if(n<5||n>maxprime)return false;U r=n%6;if(r!=1&&r!=5)return false;return bool(((r==1?A:B)[n/6/64]>>(n/6%64))&1);};
 cerr<<"sieve_seconds "<<omp_get_wtime()-begin<<"\n";
 auto get=[](const U* v,int64_t j)->U{int64_t w=j>>6;int s=j&63;return s?(v[w]>>s)|(v[w+1]<<(64-s)):v[w];};
 Stat total; U first=min(M,U(100000));
 for(U m=2;m<=first;m++){bool found=false;for(U d=0;d<=m-2;d++)if(isp(m-d)&&isp(m+d)){total.count++;total.md+=I(m-2)*d;total.dd+=I(d)*d;if(d>total.maxd){total.maxd=d;total.maxm=m;}found=true;break;}if(!found)total.miss++;}
 const int ss[6][2]={{1,5},{0,-1},{3,-1},{2,4},{3,-1},{0,-1}};
 U jstart=(first+1)/6,jend=M/6+1,step=1<<20;
 auto report=[&](U end){I L=end-2;I base=L*(L+1)*(2*L+1)/6;I den=base-2*total.md+total.dd,num=2*(total.md-total.dd);cout<<"N="<<2*end<<" count="<<total.count<<" missing_within_cap="<<total.miss<<" maxgap="<<2*total.maxd<<" maxgapN="<<2*total.maxm<<" md="<<fmt(total.md)<<" dd="<<fmt(total.dd)<<" num="<<fmt(num)<<" den="<<fmt(den)<<" slope="<<setprecision(18)<<1+(long double)num/(long double)den<<" elapsed="<<omp_get_wtime()-begin<<endl;};
 if(first==M){report(M);return 0;}
 for(U j0=jstart;j0<jend;j0+=step){U j1=min(jend,j0+step);vector<Stat> ts(omp_get_max_threads());
 #pragma omp parallel
 {Stat &st=ts[omp_get_thread_num()];
 #pragma omp for schedule(dynamic,256)
 for(U j=j0;j<j1;j+=64){for(int r=0;r<6;r++){
  U live=~U(0),base=6*j+r;
  if(base<=first){U skip=min(U(64),(first-base)/6+1);live=skip==64?0:live<<skip;}
  if(base>M)live=0;else{U len=min(U(64),(M-base)/6+1);if(len<64)live&=(U(1)<<len)-1;}
  if(!live)continue;U md=0,dd=0,localcount=0;
  for(int t=0;live&&6*t<=int(cap);t++)for(int z=0;z<2&&live;z++){
   int s=ss[r][z];if(s<0)continue;int d=6*t+s;if(d>int(cap))continue;
   int pl=r-s,ql=r+s,pc=pl<0?-1:0,qc=ql>=6?1:0,pr=(pl+6)%6,qr=ql%6;
   U hit=live&get(pr==1?A:B,int64_t(j)-t+pc)&get(qr==1?A:B,int64_t(j)+t+qc);
   if(!hit)continue;live^=hit;U cnt=0,idx=0,tmp=hit;while(tmp){idx+=__builtin_ctzll(tmp);cnt++;tmp&=tmp-1;}
   md+=U(d)*((base-2)*cnt+6*idx);dd+=U(d)*d*cnt;localcount+=cnt;if(U(d)>st.maxd){st.maxd=d;st.maxm=base+6*__builtin_ctzll(hit);}
  }
  st.md+=md;st.dd+=dd;st.count+=localcount;st.miss+=__builtin_popcountll(live);
 }} }
 for(auto &st:ts){total.md+=st.md;total.dd+=st.dd;total.count+=st.count;total.miss+=st.miss;if(st.maxd>total.maxd){total.maxd=st.maxd;total.maxm=st.maxm;}}
 U end=min(M,6*j1-1);report(end);
 }
}
