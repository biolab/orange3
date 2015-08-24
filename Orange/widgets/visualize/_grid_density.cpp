#include <map>
#include <algorithm>
using namespace std;

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif // _WIN32

int color_encode(int *rgb) {
	return rgb[0]*256*256+rgb[1]*256+rgb[2];
}

void color_decode(int c, int *rgb) {
	rgb[2]=c%256; c/=256; // b
	rgb[1]=c%256; c/=256; // g
	rgb[0]=c; // r
}

DLLEXPORT
void compute_density(int r, double *gx, double *gy, int n, double *dx, double *dy, int *drgb, int *rgba) {
	int colors = 0;
	map<int,int> color2ind, ind2color;
	int *color = new int[n];
	for (int i=0;i<n;i++) {
		int c = color_encode(drgb+i*3);
		if (color2ind.count(c)==0) {
			int ind = colors++;
			color2ind[c] = ind;
			ind2color[ind] = c;
		}
		color[i] = color2ind[c];
	}
	double min_x = *min_element(dx,dx+n), max_x = *max_element(dx,dx+n);
	double min_y = *min_element(dy,dy+n), max_y = *max_element(dy,dy+n);
	double diag = (max_x-min_x)*(max_x-min_x) + (max_y-min_y)*(max_y-min_y);
	double lo=1.0/colors, hi=1.0;
	double *f = new double[colors];
	for (int i=0;i<r;i++) {
		for (int j=0;j<r;j++) {
			double total=0;
			for (int c=0;c<colors;c++) f[c]=0;
			for (int k=0;k<n;k++) {
				int c = color[k];
				double d = 1.0/(0.001*diag + (gx[j]-dx[k])*(gx[j]-dx[k]) + (gy[i]-dy[k])*(gy[i]-dy[k]));
				total += d;
				f[c] += d;
			}
			double main_f=-1;
			int main_color=-1;
			for (int c=0;c<colors;c++) {
				if (f[c]>main_f) {
					main_f=f[c];
					main_color=c;
				}
			}
			double main_ratio=main_f/total;
			int offset=i*r*4+j*4;
			rgba[offset+3]=(lo<hi)?(int)(128*(main_ratio-lo)/(hi-lo)):128;
			color_decode(ind2color[main_color], rgba+offset);
		}
	}
	delete color;
	delete f;
}


// Empty python module definition
#include "Python.h"

static PyModuleDef _grid_density_module = {
	PyModuleDef_HEAD_INIT,
	"_grid_density",
	NULL,
	-1,
};



PyMODINIT_FUNC
PyInit__grid_density(void) {
	PyObject * mod;
	mod = PyModule_Create(&_grid_density_module);
	if (mod == NULL)
		return NULL;
	return mod;
}
