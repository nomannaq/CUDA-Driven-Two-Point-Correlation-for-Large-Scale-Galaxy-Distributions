#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int    NoofReal;
int    NoofRand;
float* real_rasc, * real_decl;
float* rand_rasc, * rand_decl;

unsigned long long int* histogramDR, * histogramDD, * histogramRR;

long int CPUMemory = 0L;
long int GPUMemory = 0L;

float totaldegrees = 360.0f;
float binsperdegree = 4.0f;
float asteriks= totaldegrees*binsperdegree;

// put here your GPU kernel(s) to calculate the histograms
__global__ void  fillHistogram(unsigned long long int* histogram, bool isDR, float* rasc1, float* rasc2, float* decl1, float* decl2, int N) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    if (i < N && j < N)
    {
     //   if (i < j || isDR) // Always compute theta for DR histogram; not always for DD,RR
      //  {
//
            float theta = sinf(decl1[i]) * sinf(decl2[j]) + cosf(decl1[i]) * cosf(decl2[j]) * cosf(rasc1[i] - rasc2[j]);
            theta = fminf(theta, 1.5f);
            theta = fmaxf(theta, -1.5f);
            theta = acosf(theta);
            //theta = acosf(fminf(fmaxf(theta, -1.0f), 1.0f));
            theta = theta * 180.0f / 3.141592654f;
            int index = isnan(theta) ? 0 : (int)(theta * 4.0f);
            atomicAdd(&histogram[index], 1);
          // int index = isnan(theta) ? 0 : (int)(theta * 4.0f);
           /* if (index >= 0 && index < 360 * 4.0f) {
                atomicAdd(&histogram[index], 1);
            }
            else {
                atomicAdd(&histogram[0], 1);
            }
            */
           // atomicAdd(&histogram[index], 1);

           /* int index = (int)(theta * 4.0f);
            if (isnan(theta)) // Handling Nan case
            {
                index = 0;omef
            }
            else
            {
                index = (int)(theta * 4.0f);
            }
            
                atomicAdd(&histogram[index], 1);*/
  
          /* if (index >= 0 && index < 360 * 4.0f)
            {
                if (isDR)
                    atomicAdd(&histogram[index], 1);
                else
                    atomicAdd(&histogram[index], 2);
            }
            else
            {
                if (isDR)
                    atomicAdd(&histogram[0], 1);
                else
                    atomicAdd(&histogram[0], 2);
            }
        }
        else if (i == j && !isDR)
        {
            atomicAdd(&histogram[index], 1);
        }*/
    }
}


int main(int argc, char* argv[])
{
    int    readdata(char* argv1, char* argv2);
    int Number_of_Galaxies = 100000;
    unsigned long long int histogramDRsum, histogramDDsum, histogramRRsum;
    int getDevice(void);

    FILE* outfil;

    if (argc != 4) { printf("Usage: a.out real_data random_data output_data\n"); return(-1); }

    clock_t walltime;
    walltime = clock();

    if (readdata(argv[1], argv[2]) != 0) return(-1);


    if (getDevice() != 0) return(-1);

    histogramDD = (unsigned long long int*)malloc(asteriks * sizeof(unsigned long long int));
    histogramRR = (unsigned long long int*)malloc(asteriks * sizeof(unsigned long long int));
    histogramDR = (unsigned long long int*)malloc(asteriks * sizeof(unsigned long long int));
    CPUMemory += 3L * (asteriks) * sizeof(unsigned long long int);

    for (int i = 0; i < asteriks; i++)
        histogramDD[i] = histogramRR[i] = histogramDR[i] = 0LLU;
    unsigned long long int* devHistogramDD, * devHistogramRR, * devHistogramDR;

    cudaMalloc(&devHistogramDD, (asteriks) * sizeof(unsigned long long int));
    cudaMalloc(&devHistogramRR, (asteriks) * sizeof(unsigned long long int));
    cudaMalloc(&devHistogramDR, (asteriks) * sizeof(unsigned long long int));

    cudaMemcpy(devHistogramDD, histogramDD, (asteriks) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(devHistogramRR, histogramRR, (asteriks) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(devHistogramDR, histogramDR, (asteriks) * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    GPUMemory += 3L * (asteriks) * sizeof(unsigned long long int);


    // Cuda Malloc/Memcpy for real_*, rand_* arrays
    float* dev_real_rasc, * dev_real_decl, * dev_rand_rasc, * dev_rand_decl;

    cudaMalloc(&dev_real_rasc, Number_of_Galaxies * sizeof(float));
    cudaMalloc(&dev_real_decl, Number_of_Galaxies * sizeof(float));
    cudaMalloc(&dev_rand_rasc, Number_of_Galaxies * sizeof(float));
    cudaMalloc(&dev_rand_decl, Number_of_Galaxies * sizeof(float));

    cudaMemcpy(dev_real_rasc, real_rasc, Number_of_Galaxies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_real_decl, real_decl, Number_of_Galaxies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rand_rasc, rand_rasc, Number_of_Galaxies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rand_decl, rand_decl, Number_of_Galaxies * sizeof(float), cudaMemcpyHostToDevice);
    GPUMemory += 4L * (Number_of_Galaxies * sizeof(float));

    // Calling GPU kernel(s) that fills the three histograms; blocksInGrid * threadsInBlock designed such that
    // every thread computes one angle and thus all 10,000,000,000 angles are computed for overall computation
    dim3 blocksInGrid(3125, 3125);
    dim3 threadsInBlock(32, 32);

    fillHistogram << < blocksInGrid, threadsInBlock >> > (devHistogramDD, false, dev_real_rasc, dev_real_rasc, dev_real_decl, dev_real_decl, Number_of_Galaxies);
    fillHistogram << < blocksInGrid, threadsInBlock >> > (devHistogramRR, false, dev_rand_rasc, dev_rand_rasc, dev_rand_decl, dev_rand_decl, Number_of_Galaxies);
    fillHistogram << < blocksInGrid, threadsInBlock >> > (devHistogramDR, true, dev_real_rasc, dev_rand_rasc, dev_real_decl, dev_rand_decl, Number_of_Galaxies);
    cudaDeviceSynchronize();

    // Copy from device to host
    cudaMemcpy(histogramDD, devHistogramDD, (asteriks) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramRR, devHistogramRR, (asteriks) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramDR, devHistogramDR, (asteriks) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

    // Free cuda memory for histograms and real_*, rand_* arrays
    cudaFree(devHistogramDD);
    cudaFree(devHistogramRR);
    cudaFree(devHistogramDR);

    cudaFree(dev_real_rasc);
    cudaFree(dev_real_decl);
    cudaFree(dev_rand_rasc);
    cudaFree(dev_rand_decl);


    // checking to see if your histograms have the right number of entries
    histogramDRsum = 0L;
    for (int i = 0; i < asteriks; ++i) {
        histogramDRsum += histogramDR[i];
    }
    printf("   DR histogram sum = %lld\n", histogramDRsum);
    if (histogramDRsum != 10000000000L) { printf("   Incorrect histogram sum, exiting..\n"); }

    histogramDDsum = 0L;
    for (int i = 0; i < asteriks; ++i)
        histogramDDsum += histogramDD[i];
    printf("   DD histogram sum = %lld\n", histogramDDsum);
    if (histogramDDsum != 10000000000L) { printf("   Incorrect histogram sum, exiting..\n"); }

    histogramRRsum = 0L;
    for (int i = 0; i < asteriks; ++i)
        histogramRRsum += histogramRR[i];
    printf("   RR histogram sum = %lld\n", histogramRRsum);
    if (histogramRRsum != 10000000000L) { printf("   Incorrect histogram sum, exiting..\n"); }


    printf("   Omega values:");

    outfil = fopen(argv[3], "w");
    if (outfil == NULL) { printf("Cannot open output file %s\n", argv[3]); return(-1); }
    fprintf(outfil, "bin start\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
    for (int i = 0; i < asteriks; ++i)
    {
        if (histogramRR[i] > 0)
        {
            double omega = (histogramDD[i] - 2 * histogramDR[i] + histogramRR[i]) / ((double)(histogramRR[i]));

            fprintf(outfil, "%6.3f\t%15lf\t%15lld\t%15lld\t%15lld\n", ((float)i) / binsperdegree, omega,
                histogramDD[i], histogramDR[i], histogramRR[i]);
            if (i < 5) printf("   %6.3lf", omega);
        }
        else
            if (i < 5) printf("         ");
    }

    printf("\n");

    fclose(outfil);

    printf("   Results written to file %s\n", argv[3]);
    printf("   CPU memory allocated  = %.2lf MB\n", CPUMemory / 1000000.0);
    printf("   GPU memory allocated  = %.2lf MB\n", GPUMemory / 1000000.0);

    walltime = clock() - walltime;

    printf("   Total wall clock time = %.2lf s\n", ((float)walltime) / CLOCKS_PER_SEC);

    return(0);
}

int readdata(char* argv1, char* argv2)
{
    int    i, linecount;
    char   inbuf[80];
    double ra, dec, dpi;
    FILE* infil;

    printf("!\n");
    // phi   = ra/60.0 * dpi/180.0;
    // theta = (90.0-dec/60.0)*dpi/180.0;
    // otherwise use 
    // phi   = ra * dpi/180.0;
    // theta = (90.0-dec)*dpi/180.0;

    dpi = 3.141592654;
    infil = fopen(argv1, "r");
    if (infil == NULL) { printf("Cannot open input file %s\n", argv1); return(-1); }

    linecount = 0;
    while (fgets(inbuf, 80, infil) != NULL) ++linecount;
    rewind(infil);

    printf("   %s contains %d galaxies\n", argv1, linecount - 1);

    NoofReal = linecount - 1;

    if (NoofReal != 100000) { printf("Incorrect number of galaxies\n"); return(1); }

    real_rasc = (float*)calloc(NoofReal, sizeof(float));
    real_decl = (float*)calloc(NoofReal, sizeof(float));
    CPUMemory += 2L * NoofReal * sizeof(float);

    fgets(inbuf, 80, infil);
    sscanf(inbuf, "%d", &linecount);
    if (linecount != 100000) { printf("Incorrect number of galaxies\n"); return(1); }

    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv1);
            fclose(infil);
            return(-1);
        }
        real_rasc[i] = (float)(ra / 60.0 * dpi / 180.0);
        real_decl[i] = (float)(dec / 60.0 * dpi / 180.0);
        ++i;
    }

    fclose(infil);

    if (i != NoofReal)
    {
        printf("   Cannot read %s correctly\n", argv1);
        return(-1);
    }

    infil = fopen(argv2, "r");
    if (infil == NULL) { printf("Cannot open input file %s\n", argv2); return(-1); }

    linecount = 0;
    while (fgets(inbuf, 80, infil) != NULL) ++linecount;
    rewind(infil);

    printf("   %s contains %d galaxies\n", argv2, linecount - 1);

    NoofRand = linecount - 1;
    if (NoofRand != 100000) { printf("Incorrect number of random galaxies\n"); return(1); }

    rand_rasc = (float*)calloc(NoofRand, sizeof(float));
    rand_decl = (float*)calloc(NoofRand, sizeof(float));
    CPUMemory += 2L * NoofRand * sizeof(float);

    fgets(inbuf, 80, infil);
    sscanf(inbuf, "%d", &linecount);
    if (linecount != 100000) { printf("Incorrect number of random galaxies\n"); return(1); }

    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv2);
            fclose(infil);
            return(-1);
        }
        rand_rasc[i] = (float)(ra / 60.0 * dpi / 180.0);
        rand_decl[i] = (float)(dec / 60.0 * dpi / 180.0);
        ++i;
    }

    fclose(infil);

    if (i != NoofReal)
    {
        printf("   Cannot read %s correctly\n", argv2);
        return(-1);
    }
    return(0);
}




int getDevice(void)
{

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("   Found %d CUDA devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128) return(-1);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("      Device %s                  device %d\n", deviceProp.name, device);
        printf("         compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem / 1000000000.0);
        printf("         l2CacheSize                  =    %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                     =    %8d\n", deviceProp.warpSize);
        printf("         clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate / 1000.0);
        printf("         maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
        printf("         asyncEngineCount             =    %8d\n", deviceProp.asyncEngineCount);
        printf("         f to lf performance ratio    =    %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
        printf("         maxGridSize                  =    %d x %d x %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("         maxThreadsDim                =    %d x %d x %d\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("         concurrentKernels            =    ");
        if (deviceProp.concurrentKernels == 1) printf("     yes\n"); else printf("    no\n");
        printf("         deviceOverlap                =    %8d\n", deviceProp.deviceOverlap);
        if (deviceProp.deviceOverlap == 1)
            printf("            Concurrently copy memory/execute kernel\n");
    }

    cudaSetDevice(0);
    cudaGetDevice(&device);
    if (device != 0) printf("   Unable to set device 0, using %d instead", device);
    else printf("   Using CUDA device %d\n\n", device);

    return(0);
}
