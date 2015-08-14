#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <set>
#include <map>
using namespace std;

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif // _WIN32

struct point {
	double x,y;
	int id, ind;
};

struct point_cmpx {
	bool operator() (const point& p1, const point& p2) const {
		return p1.x<p2.x;
	}
};

struct point_cmpy {
	bool operator() (const point& p1, const point& p2) const {
		return p1.y<p2.y;
	}
};

inline double distance(double gx, double gy, const point &p) {
	return (gx-p.x)*(gx-p.x) + (gy-p.y)*(gy-p.y);
}

void knn_line(int r, double *gx, double gy, int n, point *data, pair<double,int> *dist, int k, int *knn) {
	// find nearest neighbours of the first point
	for (int i=0;i<n;i++) dist[i] = make_pair(distance(gx[0],gy,data[i]), i);
	sort(dist, dist+n);
	for (int i=0;i<k;i++) knn[0*k+i] = dist[i].second;
	// line sweep
	multiset<point, point_cmpy> active;
	multiset<point, point_cmpy>::iterator it,lo,hi;
	int h=0, t=0;
	double s=0;
	for (int i=1;i<=r-1;i++) {
		// estimate knn distance with a distance to previous nearest neighbours
		double hint=0;
		for (int j=0;j<k;j++) {
			int cand = knn[(i-1)*k+j];
			hint = max(hint, distance(gx[i],gy,data[cand]));
		}
		hint=sqrt(hint)*(1+1e-4);
		// update left and right boarders of the active set
		while (t<h && gx[i]-data[t].x>hint) {
			active.erase(active.find(data[t]));
			t++;
		}
		while (h<n && data[h].x-gx[i]<=hint) {
			active.insert(data[h]);
			h++;
		}
		// search the relevant vertical part of the active set
		lo=active.lower_bound((point){0,gy-hint,0,0});
		hi=active.upper_bound((point){0,gy+hint,0,0});
		int a=0;
		for (it=lo; it!=hi; it++,a++) {
			dist[a]=make_pair(distance(gx[i],gy,*it), it->ind);
		}
		sort(dist, dist+a);
		s+=a;
		for (int j=0;j<k;j++) knn[i*k+j] = dist[j].second;
	}
	// replace indices with actual ids
	for (int i=0;i<r;i++) {
		for (int j=0;j<k;j++) {
			knn[i*k+j] = data[knn[i*k+j]].id;
		}
	}
}

void knn_sweep(int r, double *gx, double *gy, int n, double *dx, double *dy, int k, int *knn) {
	// sort data points horizontally
	point *data = new point[n];
	for (int i=0;i<n;i++) data[i] = (point){dx[i], dy[i], i, -1};
	sort(data, data+n, point_cmpx());
	for (int i=0;i<n;i++) data[i].ind = i;
	pair<double,int> *dist = new pair<double,int>[n];
	// find nearest neighbours of points along a horizontal line
	for (int i=0;i<r;i++) {
		knn_line(r, gx, gy[i], n, data, dist, k, knn+i*r*k);
	}
	delete data;
	delete dist;
}

void combine_colors(int k, int *knn, int *drgb, int lo, int hi, int *rgba) {
	// most represented color and its count
	map<int,int> f;
	int color_count=0;
	int main_color=0;
	for (int l=0;l<k;l++) {
		int id=knn[l];
		int c=drgb[id*3+0]*256*256+drgb[id*3+1]*256+drgb[id*3+2];
		f[c]++;
		if (f[c]>color_count) {
			color_count=f[c];
			main_color=c;
		}
	}
	rgba[3]=(lo!=hi)?(128.0*(color_count-lo)/(hi-lo)):128.0; // a
	rgba[2]=main_color%256; main_color/=256; // b
	rgba[1]=main_color%256; main_color/=256; // g
	rgba[0]=main_color; // r
}

DLLEXPORT
void compute_density(int r, double *gx, double *gy, int n, double *dx, double *dy, int *drgb, int *rgba) {
	int k = sqrt(n);
	set<int> colors;
	for (int i=0;i<n;i++) {
		colors.insert(drgb[i*3+0]*256*256+drgb[i*3+1]*256+drgb[i*3+2]);
	}
	int lo=ceil(1.0*k/colors.size()), hi=k;
	// find nearest neighbours
	int *knn = new int[r*r*k];
	knn_sweep(r,gx,gy,n,dx,dy,k,knn);
	// determine the color based on found neighbours
	map<int,int> fcc;
	for (int i=0;i<r;i++) {
		for (int j=0;j<r;j++) {
			combine_colors(k, knn+(i*r*k+j*k), drgb, lo, hi, rgba+(i*r*4+j*4));
		}
	}
	delete knn;
}
