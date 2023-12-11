#include <cufft.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
// #include "fftw3.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"

using namespace std;

void fftshift(double[], const int, const int, const int);
void absorbinglayer(double[], const int, const int, const int, double, double);
void applyboundarycondition(double[], double[], const int);
double errorcalcualtion(double[], double[], double[], double[], const int, const int);
void assignmentfntoin(double[], double[], const int);

int main()
{
  auto begin = chrono::high_resolution_clock::now();

  const int Ncn = 2048;     // 2048;
  const int Kcn = Ncn / 2;  // 1024;
  const int PML = 100;
  int NFFT = 100;
  const int NDtect = Kcn * Ncn + Kcn + 400;

  double err = 0.0001;

  double dx, dy;
  dx = 100.0 / 1000000000.0;
  dy = 100.0 / 1000000000.0;

  double pi = 3.141592653589793;
  double mu, beta, Cp, I0, a;
  a = 5.0 / 1000000.0;
  mu = 1.0;
  beta = 1.0;
  Cp = 1.0;
  I0 = 1.0;

  double f;
  double omega;
  double vf = 1500;
  double vs = 1650;

  double kf;
  double ks;

  double epsilon;

  double *shirinR, *shirinI, *shirfnR, *shirfnI;
  double *VR, *VI, *SR, *SI;
  double *GR, *GI;
  double* ABL;

  double *gammaR, *gammaI;

  double *temp11R, *temp11I, *temp22R, *temp22I;

  shirinR = (double*)malloc(sizeof(double) * Ncn * Ncn);
  shirinI = (double*)malloc(sizeof(double) * Ncn * Ncn);

  shirfnR = (double*)malloc(sizeof(double) * Ncn * Ncn);
  shirfnI = (double*)malloc(sizeof(double) * Ncn * Ncn);

  VR = (double*)malloc(sizeof(double) * Ncn * Ncn);
  VI = (double*)malloc(sizeof(double) * Ncn * Ncn);

  SR = (double*)malloc(sizeof(double) * Ncn * Ncn);
  SI = (double*)malloc(sizeof(double) * Ncn * Ncn);

  GR = (double*)malloc(sizeof(double) * Ncn * Ncn);
  GI = (double*)malloc(sizeof(double) * Ncn * Ncn);

  ABL = (double*)malloc(sizeof(double) * Ncn * Ncn);

  gammaR = (double*)malloc(sizeof(double) * Ncn * Ncn);
  gammaI = (double*)malloc(sizeof(double) * Ncn * Ncn);

  temp11R = (double*)malloc(sizeof(double) * Ncn * Ncn);
  temp11I = (double*)malloc(sizeof(double) * Ncn * Ncn);
  temp22R = (double*)malloc(sizeof(double) * Ncn * Ncn);
  temp22I = (double*)malloc(sizeof(double) * Ncn * Ncn);

  cufftDoubleComplex* fft_input = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * Ncn * Ncn);
  cufftDoubleComplex* fft_output = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * Ncn * Ncn);

  cufftDoubleComplex* fft_input_d;
  checkCudaErrors(cudaMalloc((void**)&fft_input_d, sizeof(cufftDoubleComplex) * Ncn * Ncn));
  cufftDoubleComplex* fft_output_d;
  checkCudaErrors(cudaMalloc((void**)&fft_output_d, sizeof(cufftDoubleComplex) * Ncn * Ncn));

  cufftHandle plan;
  checkCudaErrors(cufftPlan2d(&plan, Ncn, Ncn, CUFFT_Z2Z));

  int nthreads = 50;
  printf("threads=%d\n", nthreads);

  omp_set_num_threads(nthreads);

  FILE* FID12;
  FID12 = fopen("ShiCBSMultiThreadOMPSingleCell_1650.txt", "w");

  FILE* FID13;
  FID13 = fopen("TimeCBSMultiThreadOMPSingleCell_1650.txt", "w");

  double kkxy;
  int i111;

  time_t t000;
  time(&t000);
  printf("begin : %s", ctime(&t000));

  for (i111 = 1; i111 <= NFFT; i111++) {
    time_t t111;
    time(&t111);

    kkxy = 2 * pi * ((double)1 * i111) / (Ncn * dx);
    f = kkxy * vf / (2.0 * pi);
    printf("i111=%i, f=%lf\n", i111, f);

    omega = 2 * pi * f;
    kf = omega / vf;
    ks = omega / vs;
    epsilon = 0.8 * kf * kf;

// Initialization of S, V, Shi
#pragma omp parallel for
    for (int i1 = 0; i1 < Ncn; i1++) {
      for (int j1 = 0; j1 < Ncn; j1++) {
        double dist11 = sqrt((i1 - Kcn) * (i1 - Kcn) + (j1 - Kcn) * (j1 - Kcn));
        dist11 = dist11 * dx;

        int index = i1 * Ncn + j1;

        if (dist11 <= a) {
          SR[index] = 0.0;
          SI[index] = -(mu * beta * I0 * omega) / Cp;
          VR[index] = ks * ks - kf * kf;
          VI[index] = -epsilon;
        }
        else {
          SR[index] = 0.0;
          SI[index] = 0.0;
          VR[index] = 0.0;
          VI[index] = -epsilon;
        }

        shirinR[index] = 0.0;
        shirinI[index] = 0.0;
        shirfnR[index] = 0.0;
        shirfnI[index] = 0.0;

        gammaR[index] = -VI[index] / epsilon;
        gammaI[index] = VR[index] / epsilon;
      }
    }

    // initialize GR, GI
#pragma omp parallel for
    for (int i1 = 0; i1 < Ncn; i1++) {
      double ky = 2 * pi * (i1 - Kcn) / (Ncn * dx);
      for (int j1 = 0; j1 < Ncn; j1++) {
        double kx = 2 * pi * (j1 - Kcn) / (Ncn * dy);
        GR[i1 * Ncn + j1] = (kx * kx + ky * ky - kf * kf) / ((kx * kx + ky * ky - kf * kf) * (kx * kx + ky * ky - kf * kf) + epsilon * epsilon);
        GI[i1 * Ncn + j1] = epsilon / ((kx * kx + ky * ky - kf * kf) * (kx * kx + ky * ky - kf * kf) + epsilon * epsilon);
      }
    }

    fftshift(GR, Ncn, Ncn, Kcn);
    fftshift(GI, Ncn, Ncn, Kcn);

    // Assignment of initial field

#pragma omp parallel for
    for (int i1 = 0; i1 < Ncn * Ncn; i1++) {
      fft_input[i1] = make_cuDoubleComplex(SR[i1], SI[i1]);
    }

    checkCudaErrors(cudaMemcpy(fft_input_d, fft_input, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(fft_input, fft_input_d, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyDeviceToHost));
    checkCudaErrors(cufftExecZ2Z(plan, fft_input_d, fft_output_d, CUFFT_FORWARD));

    checkCudaErrors(cudaMemcpy(fft_output, fft_output_d, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyDeviceToHost));

#pragma omp parallel for
    for (int i1 = 0; i1 < Ncn * Ncn; i1++) {
      fft_input[i1] = make_cuDoubleComplex(GR[i1] * fft_output[i1].x - GI[i1] * fft_output[i1].y, GI[i1] * fft_output[i1].x + GR[i1] * fft_output[i1].y);
    }

    cudaMemcpy(fft_input_d, fft_input, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyHostToDevice);

    cufftExecZ2Z(plan, fft_input_d, fft_output_d, CUFFT_INVERSE);

    cudaMemcpy(fft_output, fft_output_d, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyDeviceToHost);

#pragma omp parallel for
    for (int i1 = 0; i1 < Ncn * Ncn; i1++) {
      shirinR[i1] = (gammaR[i1] * fft_output[i1].x - gammaI[i1] * fft_output[i1].y) / ((double)Ncn * Ncn);
      shirinI[i1] = (gammaI[i1] * fft_output[i1].x + gammaR[i1] * fft_output[i1].y) / ((double)Ncn * Ncn);
    }

    // Formation of absorbing layer
    absorbinglayer(ABL, Ncn, Kcn, PML, epsilon, dx);

    // Calculation of new field iteratively
    double Error11;
    int i222, ITEmax;

    for (i222 = 0; i222 < 2000; i222++) {
#pragma omp parallel for
      for (int i1 = 0; i1 < Ncn * Ncn; i1++) {
        fft_input[i1] = make_cuDoubleComplex(VR[i1] * shirinR[i1] - VI[i1] * shirinI[i1] + SR[i1], VI[i1] * shirinR[i1] + VR[i1] * shirinI[i1] + SI[i1]);
      }

      cudaMemcpy(fft_input_d, fft_input, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyHostToDevice);

      cufftExecZ2Z(plan, fft_input_d, fft_output_d, CUFFT_FORWARD);

      cudaMemcpy(fft_output, fft_output_d, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyDeviceToHost);

#pragma omp parallel for
      for (int i1 = 0; i1 < Ncn * Ncn; i1++) {
        fft_input[i1] = make_cuDoubleComplex(GR[i1] * fft_output[i1].x - GI[i1] * fft_output[i1].y, GI[i1] * fft_output[i1].x + GR[i1] * fft_output[i1].y);
      }

      cudaMemcpy(fft_input_d, fft_input, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyHostToDevice);
      cufftExecZ2Z(plan, fft_input_d, fft_output_d, CUFFT_INVERSE);
      cudaMemcpy(fft_output, fft_output_d, sizeof(cufftDoubleComplex) * Ncn * Ncn, cudaMemcpyDeviceToHost);

#pragma omp parallel for
      for (int i1 = 0; i1 < Ncn * Ncn; i1++) {
        temp11R[i1] = shirinR[i1] - fft_output[i1].x / ((double)Ncn * Ncn);
        temp11I[i1] = shirinI[i1] - fft_output[i1].y / ((double)Ncn * Ncn);

        temp22R[i1] = gammaR[i1] * temp11R[i1] - gammaI[i1] * temp11I[i1];
        temp22I[i1] = gammaI[i1] * temp11R[i1] + gammaR[i1] * temp11I[i1];
        shirfnR[i1] = shirinR[i1] - temp22R[i1];
        shirfnI[i1] = shirinI[i1] - temp22I[i1];
      }

      applyboundarycondition(shirfnR, ABL, Ncn);
      applyboundarycondition(shirfnI, ABL, Ncn);

      Error11 = errorcalcualtion(shirfnR, shirfnI, shirinR, shirinI, Ncn, Kcn);  //-->// error calculation

      if (Error11 <= err) {
        ITEmax = i222;
        break;
      }
      else {
        assignmentfntoin(shirinR, shirfnR, Ncn);
        assignmentfntoin(shirinI, shirfnI, Ncn);
      }
    }  //*********************----->// iteration stops

    // fftw_destroy_plan(FFT_forward);
    // fftw_destroy_plan(FFT_backward);

    /* printf("%i, %lf\n", i222, Error11);
    printf("%lf, %d, %lf, %lf, %lf, %lf\n", f, ITEmax, shirinR[NDtect], shirinI[NDtect], shirfnR[NDtect], shirfnI[NDtect]);
    fprintf(FID12, "%lf, %d, %lf, %lf, %lf, %lf\n", f, ITEmax, shirinR[NDtect], shirinI[NDtect], shirfnR[NDtect], shirfnI[NDtect]);*/
    // printf( "%lf, %lf\n", shirfnR[100], shirfnI[100]);       //----->//single value

    // fprintf(FID12, "begin : %s", ctime(&t111));
    // printf("begin : %s", ctime(&t111));

    int kk;
    for (kk = 0; kk < 1024; kk++) {
      // printf("%lf %lf\n",shirfnR[512,kk], shirfnI[512,kk]);
      // fprintf(FID12, "%lf, %d, %lf, %lf\n", f, ITEmax, shirfnR[Ncn*(Kcn-1)+kk], shirfnI[Ncn*(Kcn-1)+kk]);
      fprintf(FID12, "%lf, %d, %lf, %lf\n", f, ITEmax, shirfnR[Ncn * Kcn + kk], shirfnI[Ncn * Kcn + kk]);
      // fprintf(FID12, "%lf, %d, %lf, %lf\n", f, ITEmax, shirfnR[Ncn*Kcn+kk,kk], shirfnI[Ncn*Kcn+kk,kk]);
    }

    time_t t222;
    time(&t222);

    printf("begin : %s", ctime(&t111));
    printf("end : %s", ctime(&t222));

    fprintf(FID13, "begin : %s", ctime(&t111));
    fprintf(FID13, "end : %s", ctime(&t222));
  }

  fclose(FID12);
  fclose(FID13);

  auto end = chrono::high_resolution_clock::now();
  auto total_time = chrono::duration_cast<chrono::nanoseconds>(end - begin);

  printf("Time difference: %.6f s.\n", total_time.count() * 1e-9);

  // Export Shirfn to txt
  FILE *freal, *fimag;
  char filename[50];
  snprintf(filename, 50, "../result/Shirfn_%d_real_cuda.txt", Ncn);
  freal = fopen(filename, "w");
  snprintf(filename, 50, "../result/Shirfn_%d_imag_cuda.txt", Ncn);
  fimag = fopen(filename, "w");

  for (int i = 0; i < Ncn; i++) {
    for (int j = 0; j < Ncn; j++) {
      fprintf(freal, "%.15e", shirfnR[i * Ncn + j]);
      fprintf(fimag, "%.15e", shirfnI[i * Ncn + j]);
      if (j < Ncn - 1) {
        fprintf(freal, ",");
        fprintf(fimag, ",");
      }
    }
    if (i < Ncn - 1) {
      fprintf(freal, "\n");
      fprintf(fimag, "\n");
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void fftshift(double M[], const int N1, const int N2, const int KK)
{
  int N11, N22;
  N11 = KK;
  N22 = KK;

  double* Mtemp;
  Mtemp = (double*)malloc(sizeof(double) * N2 * N2);

#pragma omp parallel for
  for (int i1 = 0; i1 < N11; i1++) {
    for (int j1 = 0; j1 < N22; j1++) {
      long ccn11 = i1 * N2 + j1;
      long ccn22 = (i1 + N11) * N2 + (j1 + N22);
      Mtemp[ccn11] = M[ccn22];
      Mtemp[ccn22] = M[ccn11];
    }
  }

#pragma omp parallel for
  for (int i1 = N11; i1 < N1; i1++) {
    for (int j1 = 0; j1 < N22; j1++) {
      long ccn11 = i1 * N2 + j1;
      long ccn22 = (i1 - N11) * N2 + (j1 + N22);
      Mtemp[ccn11] = M[ccn22];
      Mtemp[ccn22] = M[ccn11];
    }
  }

#pragma omp parallel for
  for (int i1 = 0; i1 < N1; i1++) {
    for (int j1 = 0; j1 < N2; j1++) {
      long index = i1 * N2 + j1;
      M[index] = Mtemp[index];
    }
  }

  free(Mtemp);
}

void absorbinglayer(double mask11[], const int Ncn1, const int Kcn1, const int PML1, double epsilon1, double dx1)
{
  int i1, j1;
  long ccn;
  double r;

  // double mask11[Ncn1*Ncn1];

  // Initialize
  ccn = -1;

  // # pragma omp parallel for
  for (i1 = 0; i1 < Ncn1; i1++) {
    for (j1 = 0; j1 < Ncn1; j1++) {
      ccn = ccn + 1;
      mask11[ccn] = 1.0;
    }
  }

  // Left boundary
  ccn = -1;

  // # pragma omp parallel for
  for (i1 = 0; i1 < Ncn1; i1++) {
    for (j1 = 0; j1 < PML1; j1++) {
      ccn = i1 * Ncn1 + j1;
      r = dx1 * (sqrt((i1 - Kcn1) * (i1 - Kcn1) + (j1 - Kcn1) * (j1 - Kcn1)));
      mask11[ccn] = exp(-r * sqrt(epsilon1));
    }
  }

  // Right boundary
  ccn = -1;
  // # pragma omp parallel for
  for (i1 = 0; i1 < Ncn1; i1++) {
    for (j1 = Ncn1 - PML1; j1 < Ncn1; j1++) {
      ccn = i1 * Ncn1 + j1;
      r = dx1 * (sqrt((i1 - Kcn1) * (i1 - Kcn1) + (j1 - Kcn1) * (j1 - Kcn1)));
      mask11[ccn] = exp(-r * sqrt(epsilon1));
    }
  }

  // Upper boundary
  ccn = -1;
  // # pragma omp parallel for
  for (i1 = 0; i1 < PML1; i1++) {
    for (j1 = PML1; j1 < Ncn1 - PML1; j1++) {
      ccn = i1 * Ncn1 + j1;
      r = dx1 * (sqrt((i1 - Kcn1) * (i1 - Kcn1) + (j1 - Kcn1) * (j1 - Kcn1)));
      mask11[ccn] = exp(-r * sqrt(epsilon1));
    }
  }

  // Lower boundary
  ccn = -1;
  // # pragma omp parallel for
  for (i1 = Ncn1 - PML1; i1 < Ncn1; i1++) {
    for (j1 = PML1; j1 < Ncn1 - PML1; j1++) {
      ccn = i1 * Ncn1 + j1;
      r = dx1 * (sqrt((i1 - Kcn1) * (i1 - Kcn1) + (j1 - Kcn1) * (j1 - Kcn1)));
      mask11[ccn] = exp(-r * sqrt(epsilon1));
    }
  }
}

void applyboundarycondition(double Mfn[], double ABL1[], const int Ncn1)
{
  int i1, j1;

#pragma omp parallel for
  for (i1 = 0; i1 < Ncn1 * Ncn1; i1++) {
    Mfn[i1] = Mfn[i1] * ABL1[i1];
  }
}

double errorcalcualtion(double MR[], double MI[], double QR[], double QI[], const int Ncn1, const int Kcn1)
{
  int i1;
  double error11;

  double nume1, nume2, deno1, deno2;

  error11 = 0.0;

  double sum11 = 0.0;
  double sum22 = 0.0;

  for (i1 = 0; i1 < Ncn1; i1++) {
    nume1 = MR[Kcn1 * Ncn1 + i1];
    nume2 = MI[Kcn1 * Ncn1 + i1];

    deno1 = QR[Kcn1 * Ncn1 + i1];
    deno2 = QI[Kcn1 * Ncn1 + i1];

    sum11 = sum11 + sqrt((nume1 - deno1) * (nume1 - deno1) + (nume2 - deno2) * (nume2 - deno2));
    sum22 = sum22 + sqrt(deno1 * deno1 + deno2 * deno2);
  }

  error11 = sum11 / sum22;

  return error11;
}

void assignmentfntoin(double Min[], double Mfn[], const int Ncn1)
{
  int i1;

#pragma omp parallel for
  for (i1 = 0; i1 < Ncn1 * Ncn1; i1++) {
    Min[i1] = Mfn[i1];
  }
}
