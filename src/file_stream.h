#pragma once
#include <iostream>
#include <string>
#include <bitset>
#include <vector>
#include "bitfile.h"
#include "bitmemory.h"

#define BITSET_SIZE 2048
class FileStream
{
  public:
    void InitForRead(FILE *fpIn, bool inMemory)
    {
        FileStream::isInMemory = inMemory;
        mode = 1;
        if (inMemory)
        {
            rewind(fpIn);
            fseek(fpIn, 0, SEEK_END);
            int fileSizeInput = ftell(fpIn);
            rewind(fpIn);
            buffer = new unsigned char[fileSizeInput];
            if (fread(buffer, 1, fileSizeInput, fpIn) != fileSizeInput)
            {
                printf("Error reading input file to memory\n");
                exit(EXIT_FAILURE);
            }
            currentLength = 0;
            totalLength = fileSizeInput;
        }
        else
        {
            currentFile = fpIn;
        }
    }
    void InitForWrite(FILE *fpOut, bool inMemory, FILE *fpIn)
    {
        mode = 2;
        FileStream::isInMemory = inMemory;
        if (inMemory)
        {
            // bitMemoryRef = MakeBitMemory()
            rewind(fpIn);
            fseek(fpIn, 0, SEEK_END);
            int fileSizeInput = ftell(fpIn);
            rewind(fpIn);
            
            //currentBitset = bit_array_create(fileSizeInput * 100);
            totalLength = fileSizeInput + (fileSizeInput / 18) + 1;
            buffer = new unsigned char[totalLength];
            bitMemoryRef = MakeBitMemory(buffer,totalLength,BM_WRITE);
            currentLength = 0;
            currentFile = fpOut;
        }
        else
        {
            currentFile = fpOut;
            bitFileRef = MakeBitFile(fpOut, BF_WRITE);
        }
    }

    void InitForReadBit(FILE *fpIn, bool inMemory)
    {
        FileStream::isInMemory = inMemory;
        mode = 1;
        if (inMemory)
        {
            rewind(fpIn);
            fseek(fpIn, 0, SEEK_END);
            int fileSizeInput = ftell(fpIn);
            rewind(fpIn);
            buffer = new unsigned char[fileSizeInput];
            if (fread(buffer, 1, fileSizeInput, fpIn) != fileSizeInput)
            {
                printf("Error reading input file to memory\n");
                exit(EXIT_FAILURE);
            }
            currentLength = 0;
            totalLength = fileSizeInput;
            bitMemoryRef = MakeBitMemory(buffer,totalLength,BM_READ);
        }
        else
        {
            currentFile = fpIn;
            bitFileRef = MakeBitFile(fpIn,BF_READ);
        }
    }
    inline void PutBit(const int c)
    {
        if (!isInMemory)
        {
            BitFilePutBit(c, bitFileRef);
        }
        else
        {
            BitMemoryPutBit(c,bitMemoryRef);
            // int vectorPosition = currentLength / BITSET_SIZE;
            // int bitsetPosition = currentLength % BITSET_SIZE;
            // printf("PUTBIT %i",c);
            // if(currentLength > totalLength){
            //     printf("currentLength %i smaller than totalLength %i\n",currentLength,totalLength);
            //     exit(-1);
            // }
            // if (c == 1)
            // {
            //     bit_array_set_bit(currentBitset, currentLength++);
            // }
            // else
            // {
            //     bit_array_clear_bit(currentBitset, currentLength++);
            // }
            // (*currentBitset)[vectorPosition].set(bitsetPosition,c == 1);
            // currentLength++;
        }
    }
    inline int GetBit()
    {
        if (!isInMemory)
        {
            return BitFileGetBit(bitFileRef);
        }
        else
        {
            return BitMemoryGetBit(bitMemoryRef);
        }
    }
    inline void PutChar(const int c)
    {
        if (!isInMemory)
        {
            BitFilePutChar(c, bitFileRef);
        }
        else
        {
            BitMemoryPutChar(c,bitMemoryRef);
        }
    }
    inline int BitGetChar()
    {
        if (!isInMemory)
        {
            return BitFileGetChar(bitFileRef);
        }
        else
        {
            return BitMemoryGetChar(bitMemoryRef);
        }
    }
    inline int PutBitsNum(void *bits, const unsigned int count, const size_t size)
    {
        if (!isInMemory)
        {
            return BitFilePutBitsNum(bitFileRef, bits, count, size);
        }
        else
        {
            return BitMemoryPutBitsNum(bitMemoryRef,bits,count,size);
            
        }
    }
     inline int GetBitsNum(void *bits, const unsigned int count, const size_t size)
    {
        if (!isInMemory)
        {
            return BitFileGetBitsNum(bitFileRef, bits, count, size);
        }
        else
        {
            return BitMemoryGetBitsNum(bitMemoryRef,bits,count,size);
         
        }
    }
    inline void Flush()
    {
        if (!isInMemory)
        {
            BitFileToFILE(bitFileRef);
        }else{
            BitMemoryToArray(bitMemoryRef,&currentLength);
            // printf("Size is %i\n",currentLength);
        }
    }
    inline void WriteFile()
    {
        if (isInMemory)
        {
            rewind(currentFile);
            for(int i = 0; i < currentLength; i++){
                fputc(buffer[i],currentFile);
            }
            // bit_index_t i;
            // currentBitset->num_of_bits = currentLength;
            //         rewind(currentFile);

            // bit_array_save(currentBitset,currentFile);
            // for(i = 0; i < currentLength; i++)
            // {
            //     fprintf(currentFile, "%c", bit_array_get(currentBitset, i) ? '1' : '0');
            // }
            // auto bfpOut = MakeBitFile(currentFile, BF_WRITE);
            // for(int i = 0; i < currentLength;i++){
            //     BitFilePutBit(bit_array_get_bit(currentBitset,i)?1:0,bfpOut);
            // }
            // BitFileToFILE(bfpOut);
            
        }
    }
    inline int GetChar()
    {
        if (isInMemory)
        {
            if (currentLength < totalLength)
            {
                currentLength++;
                return buffer[currentLength - 1];
            }
            return EOF;
        }
        else
        {
            return getc(currentFile);
        }
    }

    ~FileStream()
    {
        if (isInMemory)
        {
            if (mode == 1)
            {
                delete[] buffer;
            }
            else
            {

                delete[] buffer;
                // fclose(currentFile);
                // delete currentBitset;
            }
        }
    }

  private:
    FILE *currentFile;
    unsigned char *buffer;
    bool isInMemory;
    int mode;
    int totalLength;
    int currentLength;
    bit_file_t *bitFileRef;
    bit_memory_t *bitMemoryRef;
};