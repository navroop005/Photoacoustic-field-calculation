#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include "helper_cuda.h"

using namespace std;

double errorcalcualtion(cuDoubleComplex* M, cuDoubleComplex* Q, const int Ncn1, const int Kcn1);
__global__ void fftshift_g(cuDoubleComplex* M, const int N1, const int N2, const int Kcn);
__global__ void absorbinglayer_g(double* mask11, const int Ncn1, const int Kcn1, const int PML1, double epsilon1, double dx1);
__global__ void applyboundarycondition_g(cuDoubleComplex* Mfn, double* ABL1, const int Ncn1);
__global__ void multiply_g(cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex* C, const int Ncn1);
__global__ void s_multiply_g(cuDoubleComplex* A, cuDoubleComplex B, cuDoubleComplex* C, const int Ncn1);
__global__ void sum_g(cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex* C, const int Ncn1);
__global__ void sub_g(cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex* C, const int Ncn1);
__global__ void s_divide_g(cuDoubleComplex* A, cuDoubleComplex B, cuDoubleComplex* C, const int Ncn1);

__global__ void initalize_S_V_gamma_G(cuDoubleComplex* S, cuDoubleComplex* V, cuDoubleComplex* gamma, cuDoubleComplex* G, int Ncn, int Kcn, double dx, double a, cuDoubleComplex S_in, cuDoubleComplex S_out, cuDoubleComplex V_in, cuDoubleComplex V_out, double epsilon, double kf);

int main()
{
  auto begin = chrono::high_resolution_clock::now();

  const int Ncn = 2048;
  const int Kcn = Ncn / 2;
  const int PML = 100;
  const int NFFT = 100;
  const int NDtect = Kcn * Ncn + Kcn + 400;

  double err = 0.0001;

  double dx = 100.0 / 1000000000.0;

  double a = 5.0 / 1000000.0;
  double mu = 1.0;
  double beta = 1.0;
  double Cp = 1.0;
  double I0 = 1.0;

  double f;
  double omega;
  double vf = 1500;
  double vs = 1650;

  double pi = 3.141592653589793;

  double kf;
  double ks;

  double epsilon;

  cuDoubleComplex* shirin;
  cudaMalloc((void**)&shirin, sizeof(cuDoubleComplex) * Ncn * Ncn);

  cuDoubleComplex* shirfn;
  cudaMalloc((void**)&shirfn, sizeof(cuDoubleComplex) * Ncn * Ncn);

  cuDoubleComplex* V;
  cudaMalloc((void**)&V, sizeof(cuDoubleComplex) * Ncn * Ncn);

  cuDoubleComplex* S;
  cudaMalloc((void**)&S, sizeof(cuDoubleComplex) * Ncn * Ncn);

  cuDoubleComplex* G;
  cudaMalloc((void**)&G, sizeof(cuDoubleComplex) * Ncn * Ncn);

  double* ABL;
  cudaMalloc((void**)&ABL, sizeof(double) * Ncn * Ncn);

  cuDoubleComplex* gamma;
  cudaMalloc((void**)&gamma, sizeof(cuDoubleComplex) * Ncn * Ncn);

  cufftDoubleComplex* fft_input;
  cudaMalloc((void**)&fft_input, sizeof(cufftDoubleComplex) * Ncn * Ncn);

  cufftDoubleComplex* fft_output;
  cudaMalloc((void**)&fft_output, sizeof(cufftDoubleComplex) * Ncn * Ncn);

  cufftHandle plan;
  cufftPlan2d(&plan, Ncn, Ncn, CUFFT_Z2Z);

  checkCudaErrors(cudaGetLastError());

  int BLOCK_SIZE = 16;
  int GRID_SIZE = ceil((float)Ncn / BLOCK_SIZE);

  dim3 grid(GRID_SIZE, GRID_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  FILE* FID12;
  FID12 = fopen("ShiCBSMultiThreadOMPSingleCell_1650.txt", "w");

  FILE* FID13;
  FID13 = fopen("TimeCBSMultiThreadOMPSingleCell_1650.txt", "w");

  double kkxy;
  int i111;

  time_t t000;
  time(&t000);
  printf("begin : %s", ctime(&t000));

  cuDoubleComplex* temp = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * Ncn * Ncn);

  cuDoubleComplex* shirfnMid = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * Ncn);
  cuDoubleComplex* shirinMid = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * Ncn);

  for (i111 = 1; i111 <= NFFT; i111++) {
    auto iter_begin = chrono::high_resolution_clock::now();

    kkxy = 2 * pi * ((double)1 * i111) / (Ncn * dx);
    f = kkxy * vf / (2.0 * pi);
    printf("i111=%i, f=%lf\n", i111, f);

    omega = 2 * pi * f;
    kf = omega / vf;
    ks = omega / vs;
    epsilon = 0.8 * kf * kf;

    // Initialization of S, V, Shi
    cuDoubleComplex S_in = make_cuDoubleComplex(0.0, -(mu * beta * I0 * omega) / Cp);
    cuDoubleComplex S_out = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex V_in = make_cuDoubleComplex(ks * ks - kf * kf, -epsilon);
    cuDoubleComplex V_out = make_cuDoubleComplex(0.0, -epsilon);

    initalize_S_V_gamma_G<<<grid, threads>>>(S, V, gamma, G, Ncn, Kcn, dx, a, S_in, S_out, V_in, V_out, epsilon, kf);

    cudaMemset2D(shirin, sizeof(cuDoubleComplex) * Ncn, 0, Ncn, Ncn);
    cudaMemset2D(shirfn, sizeof(cuDoubleComplex) * Ncn, 0, Ncn, Ncn);

    fftshift_g<<<grid, threads>>>(G, Ncn, Ncn, Kcn);

    cufftExecZ2Z(plan, S, fft_output, CUFFT_FORWARD);
    multiply_g<<<grid, threads>>>(fft_output, G, fft_input, Ncn);
    cufftExecZ2Z(plan, fft_input, fft_output, CUFFT_INVERSE);
    multiply_g<<<grid, threads>>>(fft_output, gamma, shirin, Ncn);
    s_divide_g<<<grid, threads>>>(shirin, make_cuDoubleComplex(Ncn * Ncn, 0), shirin, Ncn);

    absorbinglayer_g<<<grid, threads>>>(ABL, Ncn, Kcn, PML, epsilon, dx);

    checkCudaErrors(cudaGetLastError());

    // Calculation of new field iteratively
    double Error11;
    int i222, ITEmax;

    for (i222 = 0; i222 < 2000; i222++)  //--->// iteration starts
    {
      multiply_g<<<grid, threads>>>(shirin, V, fft_input, Ncn);
      sum_g<<<grid, threads>>>(fft_input, S, fft_input, Ncn);

      cufftExecZ2Z(plan, fft_input, fft_output, CUFFT_FORWARD);

      multiply_g<<<grid, threads>>>(fft_output, G, fft_input, Ncn);

      cufftExecZ2Z(plan, fft_input, fft_output, CUFFT_INVERSE);

      s_divide_g<<<grid, threads>>>(fft_output, make_cuDoubleComplex(Ncn * Ncn, 0), fft_output, Ncn);
      sub_g<<<grid, threads>>>(shirin, fft_output, fft_output, Ncn);
      multiply_g<<<grid, threads>>>(fft_output, V, fft_output, Ncn);
      s_multiply_g<<<grid, threads>>>(fft_output, make_cuDoubleComplex(0, 1 / epsilon), fft_output, Ncn);
      sub_g<<<grid, threads>>>(shirin, fft_output, shirfn, Ncn);

      applyboundarycondition_g<<<grid, threads>>>(shirfn, ABL, Ncn);

      cudaMemcpy(shirfnMid, shirfn + Kcn * Ncn, sizeof(cuDoubleComplex) * Ncn, cudaMemcpyDeviceToHost);
      cudaMemcpy(shirinMid, shirin + Kcn * Ncn, sizeof(cuDoubleComplex) * Ncn, cudaMemcpyDeviceToHost);

      Error11 = errorcalcualtion(shirfnMid, shirinMid, Ncn, Kcn);  //-->// error calculation

      if (Error11 <= err) {
        ITEmax = i222;
        break;
      }
      else {
        swap(shirin, shirfn);
      }

      // checkCudaErrors(cudaGetLastError());/
    }  //*********************----->// iteration stops

    auto iter_end = chrono::high_resolution_clock::now();
    auto iter_time = chrono::duration_cast<chrono::nanoseconds>(iter_end - iter_begin);

    printf("Saturation Iteration: %d, time: %.6f s\n\n", ITEmax, iter_time.count() * 1e-9);
  }

  fclose(FID12);
  fclose(FID13);

  auto end = chrono::high_resolution_clock::now();
  auto total_time = chrono::duration_cast<chrono::nanoseconds>(end - begin);

  printf("Time difference: %.6f s.\n", total_time.count() * 1e-9);

  // Export Shirfn to txt
  // cudaMemcpy2D(temp, sizeof(cuDoubleComplex) * Ncn, shirfn, sizeof(cuDoubleComplex) * Ncn, sizeof(cuDoubleComplex) * Ncn, Ncn, cudaMemcpyDeviceToHost);

  // FILE *freal, *fimag;
  // char filename[50];
  // snprintf(filename, 50, "../result/Shirfn_%d_real_cuda_4.txt", Ncn);
  // freal = fopen(filename, "w");
  // snprintf(filename, 50, "../result/Shirfn_%d_imag_cuda_4.txt", Ncn);
  // fimag = fopen(filename, "w");

  // for (int i = 0; i < Ncn; i++) {
  //   for (int j = 0; j < Ncn; j++) {
  //     fprintf(freal, "%.15e", temp[i * Ncn + j].x);
  //     fprintf(fimag, "%.15e", temp[i * Ncn + j].y);
  //     if (j < Ncn - 1) {
  //       fprintf(freal, ",");
  //       fprintf(fimag, ",");
  //     }
  //   }
  //   if (i < Ncn - 1) {
  //     fprintf(freal, "\n");
  //     fprintf(fimag, "\n");
  //   }
  // }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initalize_S_V_gamma_G(cuDoubleComplex* S, cuDoubleComplex* V, cuDoubleComplex* gamma, cuDoubleComplex* G, int Ncn, int Kcn, double dx, double a, cuDoubleComplex S_in, cuDoubleComplex S_out, cuDoubleComplex V_in, cuDoubleComplex V_out, double epsilon, double kf)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  double pi = 3.141592653589793;

  if (row < Ncn && column < Ncn) {
    int index = row * Ncn + column;

    double dist11 = sqrt((float)((row - Kcn) * (row - Kcn) + (column - Kcn) * (column - Kcn)));
    dist11 = dist11 * dx;

    if (dist11 <= a) {
      S[index] = S_in;
      V[index] = V_in;
    }
    else {
      S[index] = S_out;
      V[index] = V_out;
    }

    gamma[index].x = -V[index].y / epsilon;
    gamma[index].y = V[index].x / epsilon;

    double ky = 2 * pi * (row - Kcn) / (Ncn * dx);
    double kx = 2 * pi * (column - Kcn) / (Ncn * dx);

    G[index].x = (kx * kx + ky * ky - kf * kf) / ((kx * kx + ky * ky - kf * kf) * (kx * kx + ky * ky - kf * kf) + epsilon * epsilon);
    G[index].y = epsilon / ((kx * kx + ky * ky - kf * kf) * (kx * kx + ky * ky - kf * kf) + epsilon * epsilon);
  }
}

__global__ void fftshift_g(cuDoubleComplex* M, const int N1, const int N2, const int Kcn)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;

  if (i1 < N1 && j1 < N2) {
    int N11, N22;
    N11 = Kcn;
    N22 = Kcn;

    cuDoubleComplex temp;

    if (i1 < N11 && j1 < N22) {
      long ccn11 = i1 * N2 + j1;
      long ccn22 = (i1 + N11) * N2 + (j1 + N22);
      temp = M[ccn22];
      M[ccn22] = M[ccn11];
      M[ccn11] = temp;
    }
    if (i1 >= N11 && i1 < N1 && j1 < N22) {
      long ccn11 = i1 * N2 + j1;
      long ccn22 = (i1 - N11) * N2 + (j1 + N22);
      temp = M[ccn22];
      M[ccn22] = M[ccn11];
      M[ccn11] = temp;
    }
  }
}

__global__ void absorbinglayer_g(double mask11[], const int Ncn1, const int Kcn1, const int PML1, double epsilon1, double dx1)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;

  int index = i1 * Ncn1 + j1;
  mask11[index] = 1;

  double r = dx1 * (sqrt((float)((i1 - Kcn1) * (i1 - Kcn1) + (j1 - Kcn1) * (j1 - Kcn1))));
  double mask = exp(-r * sqrt(epsilon1));

  if ((i1 < Ncn1 && j1 < Ncn1) && ((i1 < Ncn1 && j1 < PML1)                                               // left
                                   || (i1 < Ncn1 && j1 >= Ncn1 - PML1 && j1 < Ncn1)                       // right
                                   || (i1 < PML1 && j1 >= PML1 && j1 < Ncn1 - PML1)                       // upper
                                   || (i1 >= Ncn1 - PML1 && i1 < Ncn1 && j1 >= PML1 && j1 < Ncn1 - PML1)  // bottom
                                   )) {
    mask11[index] = mask;
  }
}

__global__ void applyboundarycondition_g(cuDoubleComplex* Mfn, double* ABL1, const int Ncn1)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;
  int index = i1 * Ncn1 + j1;

  if (i1 < Ncn1 && j1 < Ncn1) {
    Mfn[index].x = Mfn[index].x * ABL1[index];
    Mfn[index].y = Mfn[index].y * ABL1[index];
  }
}

__global__ void multiply_g(cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex* C, const int Ncn1)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;
  int index = i1 * Ncn1 + j1;

  if (i1 < Ncn1 && j1 < Ncn1) {
    C[index] = cuCmul(A[index], B[index]);
  }
}

__global__ void s_multiply_g(cuDoubleComplex* A, cuDoubleComplex B, cuDoubleComplex* C, const int Ncn1)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;
  int index = i1 * Ncn1 + j1;

  if (i1 < Ncn1 && j1 < Ncn1) {
    C[index] = cuCmul(A[index], B);
  }
}

__global__ void s_divide_g(cuDoubleComplex* A, cuDoubleComplex B, cuDoubleComplex* C, const int Ncn1)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;
  int index = i1 * Ncn1 + j1;

  if (i1 < Ncn1 && j1 < Ncn1) {
    C[index] = cuCdiv(A[index], B);
  }
}

__global__ void sum_g(cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex* C, const int Ncn1)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;
  int index = i1 * Ncn1 + j1;

  if (i1 < Ncn1 && j1 < Ncn1) {
    C[index].x = A[index].x + B[index].x;
    C[index].y = A[index].y + B[index].y;
  }
}

__global__ void sub_g(cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex* C, const int Ncn1)
{
  int i1 = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;
  int index = i1 * Ncn1 + j1;

  if (i1 < Ncn1 && j1 < Ncn1) {
    C[index].x = A[index].x - B[index].x;
    C[index].y = A[index].y - B[index].y;
  }
}

double errorcalcualtion(cuDoubleComplex* M, cuDoubleComplex* Q, const int Ncn1, const int Kcn1)
{
  int i1;
  double error11;

  double nume1, nume2, deno1, deno2;

  error11 = 0.0;

  double sum11 = 0.0;
  double sum22 = 0.0;

  for (i1 = 0; i1 < Ncn1; i1++) {
    nume1 = M[i1].x;
    nume2 = M[i1].y;

    deno1 = Q[i1].x;
    deno2 = Q[i1].y;

    sum11 = sum11 + sqrt((nume1 - deno1) * (nume1 - deno1) + (nume2 - deno2) * (nume2 - deno2));
    sum22 = sum22 + sqrt(deno1 * deno1 + deno2 * deno2);
  }

  error11 = sum11 / sum22;

  return error11;
}
