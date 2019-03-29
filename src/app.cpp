/***************************************************************************
*                     Sample Program Using LZSS Library
*
*   File    : sample.c
*   Purpose : Demonstrate usage of LZSS library
*   Author  : Michael Dipperstein
*   Date    : February 21, 2004
*
****************************************************************************
*
* SAMPLE: Sample usage of LZSS Library
* Copyright (C) 2004, 2006, 2007, 2014 by
* Michael Dipperstein (mdipper@alumni.engr.ucsb.edu)
*
* This file is part of the lzss library.
*
* The lzss library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of the
* License, or (at your option) any later version.
*
* The lzss library is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
* General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
***************************************************************************/

/***************************************************************************
*                             INCLUDED FILES
***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <math.h>
#include "lzss.h"
#include "optlist.h"
#include "file_stream.h"
#include "statistics.h"
#include <memory>
#include <cstdlib>
#include <vector>
#include <sstream>
#include "gpu_util.h"
/***************************************************************************
*                            TYPE DEFINITIONS
***************************************************************************/
typedef enum {
    ENCODE,
    DECODE
} modes_t;


/***************************************************************************
*                                CONSTANTS
***************************************************************************/

/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/

/***************************************************************************
*                               PROTOTYPES
***************************************************************************/

/***************************************************************************
*                                FUNCTIONS
***************************************************************************/

/****************************************************************************
*   Function   : main
*   Description: This is the main function for this program, it validates
*                the command line input and, if valid, it will either
*                encode a file using the LZSS algorithm or decode a
*                file encoded with the LZSS algorithm.
*   Parameters : argc - number of parameters
*                argv - parameter list
*   Effects    : Encodes/Decodes input file
*   Returned   : 0 for success, -1 for failure.  errno will be set in the
*                event of a failure.
****************************************************************************/
int main(int argc, char *argv[])
{
    
    option_t *optList;
    option_t *thisOpt;
    FILE *fpIn;  /* pointer to open input file */
    FILE *fpOut; /* pointer to open output file */
    modes_t mode;
    ExecutionPlan plan;
    /* initialize data */
    fpIn = NULL;
    fpOut = NULL;
    mode = ENCODE;
    bool inMemory = false;
    plan = CPU_ORIGINAL;
    bool useSpar = false;
    std::vector<int> deviceIds  = {};
    /* parse command line */
    optList = GetOptList(argc, argv, "scw:mdi:o:h?p:g:b:");
    thisOpt = optList;
    int workers = 1;
    while (thisOpt != NULL)
    {
        switch (thisOpt->option)
        {
        case 'c': /* compression mode */
            mode = ENCODE;
            break;
        case 's':
            useSpar = true;
            break;

        case 'm': /* compression mode */
            inMemory = true;
            break;
        case 'd': /* decompression mode */
            mode = DECODE;
            break;

        case 'i': /* input file name */
            if (fpIn != NULL)
            {
                fprintf(stderr, "Multiple input files not allowed.\n");
                fclose(fpIn);

                if (fpOut != NULL)
                {
                    fclose(fpOut);
                }

                FreeOptList(optList);
                return -1;
            }

            /* open input file as binary */
            fpIn = fopen(thisOpt->argument, "rb");
            if (fpIn == NULL)
            {
                perror("Opening input file");

                if (fpOut != NULL)
                {
                    fclose(fpOut);
                }

                FreeOptList(optList);
                return -1;
            }
            break;
        case 'w': /* input file name */
            workers = atoi(thisOpt->argument);
            if (atoi <= 0)
            {
                fprintf(stderr, "Workers might be bigger than 0.\n");
                FreeOptList(optList);

                return -1;
            }

            break;
        case 'b': /* input file name */
            LzssBatchSize = atoi(thisOpt->argument) * pow(1024,2);//MB
            if (LzssBatchSize < 4 * pow(1024,2))
            {
                fprintf(stderr, "BatchSize might be bigger than 4MB.\n");
                FreeOptList(optList);

                return -1;
            }

            break;
        case 'g':
        {

            std::istringstream is(thisOpt->argument);

            std::string number_as_string;
            while (std::getline(is, number_as_string, ','))
            {
                deviceIds.push_back(std::stoi(number_as_string));
            }
        }
        break;
        case 'o': /* output file name */
            if (fpOut != NULL)
            {
                fprintf(stderr, "Multiple output files not allowed.\n");
                fclose(fpOut);

                if (fpIn != NULL)
                {
                    fclose(fpIn);
                }

                FreeOptList(optList);
                return -1;
            }

            /* open output file as binary */
            fpOut = fopen(thisOpt->argument, "wb");
            if (fpOut == NULL)
            {
                perror("Opening output file");

                if (fpIn != NULL)
                {
                    fclose(fpIn);
                }

                FreeOptList(optList);
                return -1;
            }
            break;

        case 'p':
            if (strcmp(thisOpt->argument, "cuda") == 0)
            {
                plan = GPU_CUDA;
            }
            else if (strcmp(thisOpt->argument, "openacc") == 0)
            {
                plan = GPU_OPENACC;
            }
            else if (strcmp(thisOpt->argument, "opencl") == 0)
            {
                plan = GPU_OPENCL;
            }
            else if (strcmp(thisOpt->argument, "cpu_sequential") == 0)
            {
                plan = CPU_SEQUENTIAL;
            }
            else if (strcmp(thisOpt->argument, "cpu_original") == 0)
            {
                plan = CPU_ORIGINAL;
            }
            else
            {
                printf("Execution Plan \"%s\" not found", thisOpt->argument);
                return -2;
            }
            break;
        case 'h':
        case '?':
            printf("Usage: %s <options>\n\n", FindFileName(argv[0]));
            printf("options:\n");
            printf("  -c : Encode input file to output file.\n");
            printf("  -d : Decode input file to output file.\n");
            printf("  -p (cuda|opencl|cpu_original|cpu_sequential): Plan of execution.\n");
            printf("  -i <filename> : Name of input file.\n");
            printf("  -o <filename> : Name of output file.\n");
            printf("  -b <batchSize> : Batch size in MB.\n");
            printf("  -s : Enable to run on SPar.\n");
            printf("  -w <workers> : Set the number of replicas per GPU for SPar.\n");
            printf("  -g <gpus separated by comma> : Set the GPUS index.\n");
            printf("  -h | ?  : Print out command line options.\n\n");
            printf("Default: %s -c -i stdin -o stdout\n",
                   FindFileName(argv[0]));

            FreeOptList(optList);
            return 0;
        }

        optList = thisOpt->next;
        free(thisOpt);
        thisOpt = optList;
    }

    /* use stdin/out if no files are provided */
    if (fpIn == NULL)
    {
        fpIn = stdin;
    }

    if (fpOut == NULL)
    {
        fpOut = stdout;
    }
    if(deviceIds.size() == 0){
        deviceIds.push_back(0);
    }
    setDeviceIds(deviceIds);
    if (deviceIds.size() == 0)
    {
        deviceIds.push_back(0);
    }
    /* we have valid parameters encode or decode */
    if (mode == ENCODE)
    {
        // using std::chrono::steady_clock;
        std::cout << "Starting..." << std::endl;
        std::cout << "Selected Plan: ";
        switch (plan)
        {
        case GPU_CUDA:
            std::cout << "GPU_CUDA";
            break;

        case CPU_SEQUENTIAL:
            std::cout << "CPU_SEQUENTIAL";
            break;

        case CPU_ORIGINAL:
            std::cout << "CPU_ORIGINAL";
            break;

        case GPU_OPENACC:
            std::cout << "GPU_OPENACC";
            break;
        case GPU_OPENCL:
            std::cout << "GPU_OPENCL";
            break;
        default:
            std::cout << "Unknown";
            break;
        }
        std::cout << std::endl;
        std::cout << "File statistics: " << (inMemory ? "memory" : "file") << std::endl;
        std::cout << "CPU Parallel: " << (useSpar ? "YES" : "NO") << std::endl;
        std::cout << "CPU Workers: " << workers << std::endl;
        std::cout << "GPU Device Ids: ";

        for (size_t i = 0; i < deviceIds.size(); i++)
        {
            if (i != 0)
            {
                std::cout << ", ";
            }
            std::cout << deviceIds[i];
        }
        std::cout << std::endl;

        auto streamIn = new FileStream;
        auto streamOut = new FileStream;

        std::unique_ptr<AppStatistics> metrics(new AppStatistics);
        streamIn->InitForRead(fpIn, inMemory);
        streamOut->InitForWrite(fpOut, inMemory, fpIn);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (plan != CPU_ORIGINAL)
        {
            if (useSpar)
            {
                if(plan != CPU_SEQUENTIAL){
                    EncodeLZSSCpuGpu(streamIn, streamOut, plan, metrics.get(), deviceIds.size());

                }else{
                    EncodeLZSSCpuGpu(streamIn, streamOut, plan, metrics.get(), workers);
                }
            }
            else
            {
                EncodeLZSSGpu(streamIn, streamOut, plan, metrics.get());
            }
        }
        else if (plan == CPU_ORIGINAL)
        {
            EncodeLZSS(streamIn, streamOut, metrics.get());
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        long totalMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        
        streamOut->WriteFile();
        delete streamOut;
        delete streamIn;

        rewind(fpIn);
        fseek(fpIn, 0, SEEK_END);
        int fileSizeInput = ftell(fpIn);

        rewind(fpOut);
        fseek(fpOut, 0, SEEK_END);
        int fileSizeOutput = ftell(fpOut);

#define MICRO_TO_SECONDS(v) v / 1000000.0
#define BYTE_TO_MB(v) v / pow(1024, 2)

        double totalSeconds = MICRO_TO_SECONDS(totalMicroseconds);
        printf("Total time: %1.4f seconds\n", totalSeconds);
        printf("Finding match: %1.6f seconds\n", MICRO_TO_SECONDS(metrics->TimeMatching));
        printf("Input file size: %1.2f MB(%i bytes)\n", BYTE_TO_MB(fileSizeInput), fileSizeInput);
        printf("Output file size: %1.2f MB(%i bytes)\n", BYTE_TO_MB(fileSizeOutput), fileSizeOutput);
        printf("Compress ratio: %3.2f%% \n", (float)fileSizeOutput / (float)fileSizeInput * 100);
        printf("Input processing: %1.2f MB/s\n", BYTE_TO_MB(fileSizeInput) / totalSeconds);
        printf("Output processing: %1.2f MB/s\n", BYTE_TO_MB((fileSizeOutput)) / totalSeconds);
#undef MICRO_TO_SECONDS
#undef BYTE_TO_MB
    }
    else
    {
        auto streamIn = new FileStream;
        streamIn->InitForReadBit(fpIn, inMemory);
        DecodeLZSS(streamIn, fpOut);
        delete streamIn;
    }
    /* remember to close files */
    fclose(fpIn);
    fclose(fpOut);
    return 0;
}
