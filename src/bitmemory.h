#pragma once

/***************************************************************************
*                             INCLUDED FILES
***************************************************************************/
#include <stdio.h>

/***************************************************************************
*                            TYPE DEFINITIONS
***************************************************************************/
typedef enum
{
    BM_READ = 0,
    BM_WRITE = 1,
    BM_APPEND= 2,
    BM_NO_MODE
} BM_MODES;

/* incomplete type to hide implementation */
struct bit_memory_t;
typedef struct bit_memory_t bit_memory_t;

/***************************************************************************
*                               PROTOTYPES
***************************************************************************/

/* open/close file */
bit_memory_t *BitMemoryOpen(unsigned char *stream,int size, const BM_MODES mode);
bit_memory_t *MakeBitMemory(unsigned char *stream, int size, const BM_MODES mode);
int BitMemoryClose(bit_memory_t *stream);
unsigned char *BitMemoryToArray(bit_memory_t *stream, int* size);

/* toss spare bits and byte align file */
int BitMemoryByteAlign(bit_memory_t *stream);

/* fill byte with ones or zeros and write out results */
int BitMemoryFlushOutput(bit_memory_t *stream, const unsigned char onesFill);

/* get/put character */
int BitMemoryGetChar(bit_memory_t *stream);
int BitMemoryPutChar(const int c, bit_memory_t *stream);

/* get/put single bit */
int BitMemoryGetBit(bit_memory_t *stream);
int BitMemoryPutBit(const int c, bit_memory_t *stream);

/* get/put number of bits (most significant bit to least significat bit) */
int BitMemoryGetBits(bit_memory_t *stream, void *bits, const unsigned int count);
int BitMemoryPutBits(bit_memory_t *stream, void *bits, const unsigned int count);

/***************************************************************************
* get/put a number of bits from numerical types (short, int, long, ...)
*
* For these functions, the first bit in/out is the least significant bit of
* the least significant byte, so machine endiness is accounted for.  Only
* big endian and little endian architectures are currently supported.
*
* NOTE: size is the sizeof() for the data structure pointed to by bits.
***************************************************************************/
int BitMemoryGetBitsNum(bit_memory_t *stream, void *bits, const unsigned int count,
    const size_t size);
int BitMemoryPutBitsNum(bit_memory_t *stream, void *bits, const unsigned int count,
    const size_t size);

