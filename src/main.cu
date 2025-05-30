#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "CudaArray.cuh"
#include "ticktock.h"
#include "vdbExporter.h"
#include <thread>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iostream>

__global__ void advect_kernel(CudaTextureAccessor<float4> texVel, CudaSurfaceAccessor<float4> sufloc, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    auto sample = [] (CudaTextureAccessor<float4> tex, float3 location) -> float3 {
        float4 velocity = tex.sample(location.x, location.y, location.z);
        return make_float3(velocity.x, velocity.y, velocity.z);
    };

    float3 location = make_float3(x + 0.5f, y + 0.5f, z + 0.5f);
    if (sufBound.read(x, y, z) >= 0) {
        float3 k1 = sample(texVel, location);
        float3 k2 = sample(texVel, location - 0.5f * k1);
        float3 k3 = sample(texVel, location - 0.5f * k2);
        float3 k4 = sample(texVel, location - k3);
        location -= (k1 + 2.f * k2 + 2.f * k3 + k4) / 6.f;
    }
    sufloc.write(make_float4(location.x, location.y, location.z, 0.f), x, y, z);
}

template <class T>
__global__ void resample_kernel(CudaSurfaceAccessor<float4> sufloc, CudaTextureAccessor<T> texClr, CudaSurfaceAccessor<T> sufClrNext, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    float4 location = sufloc.read(x, y, z);
    T color = texClr.sample(location.x, location.y, location.z);
    sufClrNext.write(color, x, y, z);
}

__global__ void decay_kernel(CudaSurfaceAccessor<float> sufTmp, CudaSurfaceAccessor<float> sufTmpNext, CudaSurfaceAccessor<char> sufBound, float decayRate, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;
    if (sufBound.read(x, y, z) < 0) return;

    float txp = sufTmp.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float typ = sufTmp.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float tzp = sufTmp.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float txn = sufTmp.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float tyn = sufTmp.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float tzn = sufTmp.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float tmpAvg = (txp + typ + tzp + txn + tyn + tzn) * (1 / 6.f);
    float temperatureNext = sufTmp.read(x, y, z);
    temperatureNext = temperatureNext * decayRate + tmpAvg * (1.f - decayRate);
    sufTmpNext.write(temperatureNext, x, y, z);
}

__global__ void divergence_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> sufDiv, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;
    if (sufBound.read(x, y, z) < 0) {
        sufDiv.write(0.f, x, y, z);
        return;
    }

    float vxp = sufVel.read<cudaBoundaryModeClamp>(x + 1, y, z).x;
    float vyp = sufVel.read<cudaBoundaryModeClamp>(x, y + 1, z).y;
    float vzp = sufVel.read<cudaBoundaryModeClamp>(x, y, z + 1).z;
    float vxn = sufVel.read<cudaBoundaryModeClamp>(x - 1, y, z).x;
    float vyn = sufVel.read<cudaBoundaryModeClamp>(x, y - 1, z).y;
    float vzn = sufVel.read<cudaBoundaryModeClamp>(x, y, z - 1).z;
    float divergence = (vxp - vxn + vyp - vyn + vzp - vzn) * 0.5f;
    sufDiv.write(divergence, x, y, z);
}

__global__ void sumloss_kernel(CudaSurfaceAccessor<float> sufDiv, float *sum, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    float divergence = sufDiv.read(x, y, z);
    atomicAdd(sum, divergence * divergence);
}

__global__ void jacobi_kernel(CudaSurfaceAccessor<float> sufDiv, CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;
    if (sufBound.read(x, y, z) < 0) return;

    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float divergence = sufDiv.read(x, y, z);
    float pressureNext = (pxp + pxn + pyp + pyn + pzp + pzn - divergence) * (1.f / 6.f);
    sufPreNext.write(pressureNext, x, y, z);
}

__global__ void subgradient_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;
    if (sufBound.read(x, y, z) < 0) return;

    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float4 velocity = sufVel.read(x, y, z);
    velocity.x -= (pxp - pxn) * 0.5f;
    velocity.y -= (pyp - pyn) * 0.5f;
    velocity.z -= (pzp - pzn) * 0.5f;
    sufVel.write(velocity, x, y, z);
}

template <int phase>
__global__ void rbgs_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;
    if ((x + y + z) % 2 != phase) return;

    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float divergence = sufDiv.read(x, y, z);
    float pressureNext = (pxp + pxn + pyp + pyn + pzp + pzn - divergence) * (1.f / 6.f);
    sufPre.write(pressureNext, x, y, z);
}

__global__ void residual_kernel(CudaSurfaceAccessor<float> sufRes, CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float pressure = sufPre.read(x, y, z);
    float divergence = sufDiv.read(x, y, z);
    float residual = pxp + pxn + pyp + pyn + pzp + pzn - 6.f * pressure - divergence;
    sufRes.write(residual, x, y, z);
}

__global__ void restrict_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    float ooo = sufPre.read<cudaBoundaryModeClamp>(x*2, y*2, z*2);
    float ioo = sufPre.read<cudaBoundaryModeClamp>(x*2+1, y*2, z*2);
    float oio = sufPre.read<cudaBoundaryModeClamp>(x*2, y*2+1, z*2);
    float iio = sufPre.read<cudaBoundaryModeClamp>(x*2+1, y*2+1, z*2);
    float ooi = sufPre.read<cudaBoundaryModeClamp>(x*2, y*2, z*2+1);
    float ioi = sufPre.read<cudaBoundaryModeClamp>(x*2+1, y*2, z*2+1);
    float oii = sufPre.read<cudaBoundaryModeClamp>(x*2, y*2+1, z*2+1);
    float iii = sufPre.read<cudaBoundaryModeClamp>(x*2+1, y*2+1, z*2+1);
    float pressureNext = (ooo + ioo + oio + iio + ooi + ioi + oii + iii);
    sufPreNext.write(pressureNext, x, y, z);
}

__global__ void fillzero_kernel(CudaSurfaceAccessor<float> sufPre, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    sufPre.write(0.f, x, y, z);
}

__global__ void prolongate_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    float preDelta = sufPre.read(x, y, z) * (0.5f / 8.f);
#pragma unroll
    for (int dz = 0; dz < 2; dz++) {
#pragma unroll
        for (int dy = 0; dy < 2; dy++) {
#pragma unroll
            for (int dx = 0; dx < 2; dx++) {
                float pressureNext = sufPreNext.read<cudaBoundaryModeZero>(x*2+dx, y*2+dy, z*2+dz);
                pressureNext += preDelta;
                sufPreNext.write<cudaBoundaryModeZero>(pressureNext, x*2+dx, y*2+dy, z*2+dz);
            }
        }
    }
}

struct SmokeSim : DisableCopy {
    unsigned int n;
    std::unique_ptr<CudaSurface<float4>> location;
    std::unique_ptr<CudaTexture<float4>> velocity;
    std::unique_ptr<CudaTexture<float4>> velocityNext;
    std::unique_ptr<CudaTexture<float>> color;
    std::unique_ptr<CudaTexture<float>> colorNext;
    std::unique_ptr<CudaTexture<float>> temperature;
    std::unique_ptr<CudaTexture<float>> temperatureNext;

    std::unique_ptr<CudaSurface<char>> boundary;
    std::unique_ptr<CudaSurface<float>> divergence;
    std::unique_ptr<CudaSurface<float>> pressure;
    //std::unique_ptr<CudaSurface<float>> pressureNext;
    std::vector<std::unique_ptr<CudaSurface<float>>> residual;
    std::vector<std::unique_ptr<CudaSurface<float>>> residualNext;
    std::vector<std::unique_ptr<CudaSurface<float>>> errorNext;
    std::vector<unsigned int> sizes;

    explicit SmokeSim(unsigned int _n, unsigned int _n0 = 16)
    : n(_n)
    , location(std::make_unique<CudaSurface<float4>>(uint3{n, n, n}))
    , velocity(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
    , velocityNext(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
    , color(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
    , colorNext(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
    , temperature(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
    , temperatureNext(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
    , divergence(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    , pressure(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    //, pressureNext(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    , boundary(std::make_unique<CudaSurface<char>>(uint3{n, n, n}))
    {
        fillzero_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pressure->accessSurface(), n);

        unsigned int tn;
        for (tn = n; tn >= _n0; tn /= 2) {
            residual.push_back(std::make_unique<CudaSurface<float>>(uint3{tn, tn, tn}));
            residualNext.push_back(std::make_unique<CudaSurface<float>>(uint3{tn/2, tn/2, tn/2}));
            errorNext.push_back(std::make_unique<CudaSurface<float>>(uint3{tn/2, tn/2, tn/2}));
            sizes.push_back(tn);
        }
    }

    void smooth(CudaSurface<float> *v, CudaSurface<float> *f, unsigned int lev, int times = 4) {
        unsigned int tn = sizes[lev];
        for (int step = 0; step < times; step++) {
            rbgs_kernel<0><<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(v->accessSurface(), f->accessSurface(), tn);
            rbgs_kernel<1><<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(v->accessSurface(), f->accessSurface(), tn);
        }
    }

    void vcycle(unsigned int lev, CudaSurface<float> *v, CudaSurface<float> *f) {
        if (lev >= sizes.size()) {
            unsigned int tn = sizes.back() / 2;
            smooth(v, f, lev);
            return;
        }
        auto *r = residual[lev].get();
        auto *r2 = residualNext[lev].get();
        auto *e2 = errorNext[lev].get();
        unsigned int tn = sizes[lev];
        smooth(v, f, lev);
        residual_kernel<<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(r->accessSurface(), v->accessSurface(), f->accessSurface(), tn);
        restrict_kernel<<<dim3((tn/2 + 7) / 8, (tn/2 + 7) / 8, (tn/2 + 7) / 8), dim3(8, 8, 8)>>>(r2->accessSurface(), r->accessSurface(), tn/2);
        fillzero_kernel<<<dim3((tn/2 + 7) / 8, (tn/2 + 7) / 8, (tn/2 + 7) / 8), dim3(8, 8, 8)>>>(e2->accessSurface(), tn/2);
        vcycle(lev + 1, e2, r2);
        prolongate_kernel<<<dim3((tn/2 + 7) / 8, (tn/2 + 7) / 8, (tn/2 + 7) / 8), dim3(8, 8, 8)>>>(v->accessSurface(), e2->accessSurface(), tn/2);
        smooth(v, f, lev);
    }

    void projection() {
        divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(velocity->accessSurface(), divergence->accessSurface(), boundary->accessSurface(), n);
        // vcycle(0, pressure.get(), divergence.get());
        for (int i = 0; i < 4; ++i) {
            vcycle(0, pressure.get(), divergence.get());
            float residual = calc_residual();
            printf(" vcycle %d, residual = %e\n", i, residual);
        }
        /*
        for (int i = 0; i < 50; ++i) {
           vcycle(0, pressure.get(), divergence.get());
           if (i % 5 == 0 || i == 50 - 1) {
               float residual = calc_residual();
               printf("  vcycle %d, residual = %e\n", i, residual);
           }
        }
        */
        subgradient_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pressure->accessSurface(), velocity->accessSurface(), boundary->accessSurface(), n);
    }

    void advection() {
        advect_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(velocity->accessTexture(), location->accessSurface(), boundary->accessSurface(), n);

        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(location->accessSurface(), velocity->accessTexture(), velocityNext->accessSurface(), n);
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(location->accessSurface(), color->accessTexture(), colorNext->accessSurface(), n);
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(location->accessSurface(), temperature->accessTexture(), temperatureNext->accessSurface(), n);
        std::swap(velocity, velocityNext);
        std::swap(color, colorNext);
        std::swap(temperature, temperatureNext);

        decay_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(temperature->accessSurface(), temperatureNext->accessSurface(), boundary->accessSurface(), std::exp(-0.4f), n);
        std::swap(temperature, temperatureNext);
    }

    void step(int times = 16) {
        for (int step = 0; step < times; step++) {
            projection();
            // rbgs_projection();
            // jacobi_projection();
            advection();
        }
    }

    void rbgs_projection(int times = 50) {
       divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(velocity->accessSurface(), divergence->accessSurface(), boundary->accessSurface(), n);
       fillzero_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pressure->accessSurface(), n);

       for (int i = 0; i < times; ++i) {
           smooth(pressure.get(), divergence.get(), 0, 1);
           if (i % 5 == 0 || i == times - 1) {
               float residual = calc_residual();
               printf("  rbgs %d, residual = %e\n", i, residual);
           }
       }
       subgradient_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pressure->accessSurface(), velocity->accessSurface(), boundary->accessSurface(), n);
    }

    void jacobi_projection(int times = 50) {
       divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(velocity->accessSurface(), divergence->accessSurface(), boundary->accessSurface(), n);
       auto pressureNext = std::make_unique<CudaSurface<float>>(uint3{n, n, n});
       fillzero_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pressure->accessSurface(), n);

       for (int i = 0; i < times; ++i) {
           jacobi_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
               divergence->accessSurface(), pressure->accessSurface(), pressureNext->accessSurface(), boundary->accessSurface(), n);
           std::swap(pressure, pressureNext);
           if (i % 5 == 0 || i == times - 1) {
               float residual = calc_residual();
               printf("  jacobi %d, residual = %e\n", i, residual);
           }
       }
       subgradient_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pressure->accessSurface(), velocity->accessSurface(), boundary->accessSurface(), n);
    }   


    float calc_loss() {
        divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(velocity->accessSurface(), divergence->accessSurface(), boundary->accessSurface(), n);
        float *sum;
        checkCudaErrors(cudaMalloc(&sum, sizeof(float)));
        sumloss_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(divergence->accessSurface(), sum, n);
        float cpu;
        checkCudaErrors(cudaMemcpy(&cpu, sum, sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(sum));
        return cpu;
    }

    float calc_residual() {
        // residual compute r = A(pressure) - divergence
        residual_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(residual[0]->accessSurface(), pressure->accessSurface(), divergence->accessSurface(), n);
        float *sum;
        checkCudaErrors(cudaMalloc(&sum, sizeof(float)));
        sumloss_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(residual[0]->accessSurface(), sum, n);
        float cpu;
        checkCudaErrors(cudaMemcpy(&cpu, sum, sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(sum));
        return cpu;
    }
};

void write_vtk_scalar(const std::string& filename, const float* data, int n, const char* name = "density") {
    std::ofstream fout(filename);
    fout << "# vtk DataFile Version 3.0\n";
    fout << name << "\nASCII\nDATASET STRUCTURED_POINTS\n";
    fout << "DIMENSIONS " << n << " " << n << " " << n << "\n";
    fout << "ORIGIN 0 0 0\nSPACING 1 1 1\n";
    fout << "POINT_DATA " << n*n*n << "\n";
    fout << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < n*n*n; ++i) fout << data[i] << "\n";
    fout.close();
}

int main(int argc, char** argv) {

    std::string proj_method = "vcycle";
    unsigned int n = 128;
    int step_times = 16;

    if (argc > 1) proj_method = argv[1];
    if (argc > 2) n = std::atoi(argv[2]);
    if (argc > 3) step_times = std::atoi(argv[3]);

    printf("Projection: %s, Grid: %u, Step times: %d\n", proj_method.c_str(), n, step_times);

    SmokeSim sim(n);

    {
       std::vector<char> cpu(n * n * n);
       for (int z = 0; z < n; z++) {
           for (int y = 0; y < n; y++) {
               for (int x = 0; x < n; x++) {
                   char sdf1 = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? -1 : 1;
                   char sdf2 = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n * 3 / 4) < n / 6 ? -1 : 1;
                   cpu[x + n * (y + n * z)] = std::min(sdf1, sdf2); // 恢复为两个球体的并集
               }
           }
       }
       sim.boundary->copyIn(cpu.data());
    }

    {
        std::vector<float> cpu(n * n * n);
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    float den = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? 1.f : 0.f;
                    cpu[x + n * (y + n * z)] = den;
                }
            }
        }
        sim.color->copyIn(cpu.data());
    }

    {
        std::vector<float> cpu(n * n * n);
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    float temperature = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? 1.f : 0.f;
                    cpu[x + n * (y + n * z)] = temperature;
                }
            }
        }
        sim.temperature->copyIn(cpu.data());
    }

    {
        std::vector<float4> cpu(n * n * n);
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    float velocity = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? 0.9f : 0.f;
                    cpu[x + n * (y + n * z)] = make_float4(0.f, 0.f, velocity * 0.1f, 0.f);
                }
            }
        }
        sim.velocity->copyIn(cpu.data());
    }

    std::vector<std::thread> tpool;
    for (int frame = 1; frame <= 100; frame++) {
        std::vector<float> cpuClr(n * n * n);
        std::vector<float> cpuTmp(n * n * n);
        sim.color->copyOut(cpuClr.data());
        sim.temperature->copyOut(cpuTmp.data());
        tpool.push_back(std::thread([cpuClr = std::move(cpuClr), cpuTmp = std::move(cpuTmp), frame, n] {
            VDBExporter exporter;
            exporter.addGrid<float, 1>("density", cpuClr.data(), n, n, n);
            exporter.addGrid<float, 1>("temperature", cpuTmp.data(), n, n, n);
            exporter.write("smoke" + std::to_string(1000 + frame).substr(1) + ".vdb");

            write_vtk_scalar("smoke_density_" + std::to_string(frame) + ".vtk", cpuClr.data(), n, "density");
            write_vtk_scalar("smoke_temperature_" + std::to_string(frame) + ".vtk", cpuTmp.data(), n, "temperature");
        }));

        printf("frame=%d, loss=%f\n", frame, sim.calc_loss());

        if (proj_method == "vcycle") {
            sim.step(step_times);
        } else if (proj_method == "rbgs") {
            for (int i = 0; i < step_times; ++i) sim.rbgs_projection();
            sim.advection();
        } else if (proj_method == "jacobi") {
            for (int i = 0; i < step_times; ++i) sim.jacobi_projection();
            sim.advection();
        } else {
            printf("Unknown projection method: %s\n", proj_method.c_str());
            break;
        }
    }

    for (auto &t: tpool) t.join();
    return 0;
}
