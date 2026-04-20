#include <cstdio>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main()
{
    thrust::device_vector<int> d(8);
    int h[8] = {5, 2, 7, 1, 8, 3, 6, 4};
    thrust::copy(h, h + 8, d.begin());
    thrust::sort(d.begin(), d.end());
    thrust::copy(d.begin(), d.end(), h);
    std::printf("sorted:");
    for (int i = 0; i < 8; ++i) std::printf(" %d", h[i]);
    std::printf("\n");
    return 0;
}
