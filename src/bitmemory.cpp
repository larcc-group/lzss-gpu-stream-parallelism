/***************************************************************************
*                        Bit Stream Memory Implementation
*
*   Memory    : bitfile.c
*   Purpose : This file implements a simple library of I/O functions for
*             files that contain data in sizes that aren't integral bytes.
*             An attempt was made to make the functions in this library
*             analogous to functions provided to manipulate byte streams.
*             The functions contained in this library were created with
*             compression algorithms in mind, but may be suited to other
*             applications.
*   Author  : Michael Dipperstein
*   Date    : January 9, 2004
*
****************************************************************************
*
* Bitfile: Bit stream Memory I/O Routines
* Copyright (C) 2004-2014 by Michael Dipperstein (mdipper@alumni.cs.ucsb.edu)
*
* This file is part of the bit file library.
*
* The bit file library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of the
* License, or (at your option) any later version.
*
* The bit file library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
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
#include <stdlib.h>
#include <errno.h>
#include "bitmemory.h"

/***************************************************************************
*                            TYPE DEFINITIONS
***************************************************************************/

/***************************************************************************
* type to point to the kind of functions that put/get bits from/to numerical
* data types (short, int, long, ...)
* parameters: file pointer, data structure, number of bits, sizeof data.
***************************************************************************/
typedef int (*num_func_t)(bit_memory_t*, void*, const unsigned int, const size_t);

struct bit_memory_t
{
    unsigned char *fp;                   /* file pointer used by stdio functions */
    unsigned char bitBuffer;    /* bits waiting to be read/written */
    unsigned char bitCount;     /* number of bits in bitBuffer */
    int currentCount;
    int size;
    num_func_t PutBitsNumFunc;  /* endian specific BitMemoryPutBitsNum */
    num_func_t GetBitsNumFunc;  /* endian specific BitMemoryGetBitsNum */
    BM_MODES mode;              /* open for read, write, or append */
};

typedef enum
{
    BF_UNKNOWN_ENDIAN,
    BF_LITTLE_ENDIAN,
    BF_BIG_ENDIAN
} endian_t;

/* union used to test for endianess */
typedef union
{
    unsigned long word;
    unsigned char bytes[sizeof(unsigned long)];
} endian_test_t;

/***************************************************************************
*                               PROTOTYPES
***************************************************************************/
static endian_t DetermineEndianess(void);

static int BitMemoryPutBitsLE(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size);
static int BitMemoryPutBitsBE(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size);

static int BitMemoryGetBitsLE(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size);
static int BitMemoryGetBitsBE(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size);
static int BitMemoryNotSupported(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size);

/***************************************************************************
*                                FUNCTIONS
***************************************************************************/

/***************************************************************************
*   Function   : BitMemoryOpen
*   Description: This function opens a bit file for reading, writing,
*                or appending.  If successful, a bit_memory_t data
*                structure will be allocated and a pointer to the
*                structure will be returned.
*   Parameters : fileName - NULL terminated string containing the name of
*                           the file to be opened.
*                mode - The mode of the file to be opened
*   Effects    : The specified file will be opened and file structure will
*                be allocated.
*   Returned   : Pointer to the bit_memory_t structure for the bit file
*                opened, or NULL on failure.  errno will be set for all
*                failure cases.
***************************************************************************/
bit_memory_t *BitMemoryOpen(unsigned char *stream, int size, const BM_MODES mode)
{
    bit_memory_t *bf;

    bf = (bit_memory_t *)malloc(sizeof(bit_memory_t));

    if (bf == NULL)
    {
        /* malloc failed */
        errno = ENOMEM;
    }
    else
    {
        bf->fp = stream;

        if (bf->fp == NULL)
        {
            /* fopen failed */
            free(bf);
            bf = NULL;
        }
        else
        {
            /* fopen succeeded fill in remaining bf data */
            bf->bitBuffer = 0;
            bf->bitCount = 0;
            bf->mode = mode;
            bf->currentCount = 0;
            bf->size = size;

            switch (DetermineEndianess())
            {
                case BF_LITTLE_ENDIAN:
                    bf->PutBitsNumFunc = &BitMemoryPutBitsLE;
                    bf->GetBitsNumFunc = &BitMemoryGetBitsLE;
                    break;

                case BF_BIG_ENDIAN:
                    bf->PutBitsNumFunc = &BitMemoryPutBitsBE;
                    bf->GetBitsNumFunc = &BitMemoryGetBitsBE;
                    break;

                case BF_UNKNOWN_ENDIAN:
                default:
                    bf->PutBitsNumFunc = BitMemoryNotSupported;
                    bf->GetBitsNumFunc = BitMemoryNotSupported;
                    break;
            }

            /***************************************************************
            * TO DO: Consider using the last byte in a file to indicate
            * the number of bits in the previous byte that actually have
            * data.  If I do that, I'll need special handling of files
            * opened with a mode of BF_APPEND.
            ***************************************************************/
        }
    }

    return (bf);
}

/***************************************************************************
*   Function   : MakeBitMemory
*   Description: This function naively wraps a standard file in a
*                bit_memory_t structure.  ANSI-C doesn't support file status
*                functions commonly found in other C variants, so the
*                caller must be passed as a parameter.
*   Parameters : stream - pointer to the standard file being wrapped.
*                mode - The mode of the file being wrapped.
*   Effects    : A bit_memory_t structure will be created for the stream
*                passed as a parameter.
*   Returned   : Pointer to the bit_memory_t structure for the bit file
*                or NULL on failure.  errno will be set for all failure
*                cases.
***************************************************************************/
bit_memory_t *MakeBitMemory(unsigned char *stream, int size, const BM_MODES mode)
{
    bit_memory_t *bf;

    if (stream == NULL)
    {
        /* can't wrapper empty steam */
        errno = EBADF;
        bf = NULL;
    }
    else
    {
        bf = (bit_memory_t *)malloc(sizeof(bit_memory_t));

        if (bf == NULL)
        {
            /* malloc failed */
            errno = ENOMEM;
        }
        else
        {
            /* set structure data */
            bf->fp = stream;
            bf->bitBuffer = 0;
            bf->bitCount = 0;
            bf->mode = mode;

            bf->currentCount = 0;
            bf->size = size;

            switch (DetermineEndianess())
            {
                case BF_LITTLE_ENDIAN:
                    bf->PutBitsNumFunc = &BitMemoryPutBitsLE;
                    bf->GetBitsNumFunc = &BitMemoryGetBitsLE;
                    break;

                case BF_BIG_ENDIAN:
                    bf->PutBitsNumFunc = &BitMemoryPutBitsBE;
                    bf->GetBitsNumFunc = &BitMemoryGetBitsBE;
                    break;

                case BF_UNKNOWN_ENDIAN:
                default:
                    bf->PutBitsNumFunc = BitMemoryNotSupported;
                    bf->GetBitsNumFunc = BitMemoryNotSupported;
                    break;
            }
        }
    }

    return (bf);
}

/***************************************************************************
*   Function   : DetermineEndianess
*   Description: This function determines the endianess of the current
*                hardware architecture.  An unsigned long is set to 1.  If
*                the 1st byte of the unsigned long gets the 1, this is a
*                little endian machine.  If the last byte gets the 1, this
*                is a big endian machine.
*   Parameters : None
*   Effects    : None
*   Returned   : endian_t for current machine architecture
***************************************************************************/
static endian_t DetermineEndianess(void)
{
    endian_t endian;
    endian_test_t endianTest;

    endianTest.word = 1;

    if (endianTest.bytes[0] == 1)
    {
        /* LSB is 1st byte (little endian)*/
        endian = BF_LITTLE_ENDIAN;
    }
    else if (endianTest.bytes[sizeof(unsigned long) - 1] == 1)
    {
        /* LSB is last byte (big endian)*/
        endian = BF_BIG_ENDIAN;
    }
    else
    {
        endian = BF_UNKNOWN_ENDIAN;
    }

    return endian;
}

/***************************************************************************
*   Function   : BitMemoryClose
*   Description: This function closes a bit file and frees all associated
*                data.
*   Parameters : stream - pointer to bit file stream being closed
*   Effects    : The specified file will be closed and the file structure
*                will be freed.
*   Returned   : 0 for success or EOF for failure.
***************************************************************************/
int BitMemoryClose(bit_memory_t *stream)
{
    int returnValue = 0;

    if (stream == NULL)
    {
        return(EOF);
    }

    if ((stream->mode == BM_WRITE) || (stream->mode == BM_APPEND))
    {
        /* write out any unwritten bits */
        if (stream->bitCount != 0)
        {
            (stream->bitBuffer) <<= 8 - (stream->bitCount);
            if(stream->currentCount >= stream->size){
                return EOF;
            }
            stream->fp[stream->currentCount++] = stream->bitBuffer;
            // fputc(stream->bitBuffer, stream->fp);   /* handle error? */
        }
    }

    /***********************************************************************
    *  TO DO: Consider writing an additional byte indicating the number of
    *  valid bits (bitCount) in the previous byte.
    ***********************************************************************/

    /* close file */
    returnValue = 0;
    // returnValue = fclose(stream->fp);

    /* free memory allocated for bit file */
    free(stream);

    return(returnValue);
}

/***************************************************************************
*   Function   : BitMemoryToFILE
*   Description: This function flushes and frees the bitfile structure,
*                returning a pointer to a stdio file.
*   Parameters : stream - pointer to bit file stream being closed
*   Effects    : The specified bitfile will be made usable as a stdio
*                FILE.
*   Returned   : Pointer to FILE.  NULL for failure.
***************************************************************************/
unsigned char *BitMemoryToArray(bit_memory_t *stream, int* bitArraySize)
{
    unsigned char * fp;
    if (stream == NULL)
    {
        return(NULL);
    }

    if ((stream->mode == BM_WRITE) || (stream->mode == BM_APPEND))
    {
        /* write out any unwritten bits */
        if (stream->bitCount != 0)
        {
            (stream->bitBuffer) <<= 8 - (stream->bitCount);
            if(stream->currentCount >= stream->size){
                return NULL;
            }
            stream->fp[stream->currentCount++] = stream->bitBuffer;
            // fputc(stream->bitBuffer, stream->fp);   /* handle error? */
        }
    }

    /***********************************************************************
    *  TO DO: Consider writing an additional byte indicating the number of
    *  valid bits (bitCount) in the previous byte.
    ***********************************************************************/

    /* close file */
    fp = stream->fp;    
    (*bitArraySize) = stream->currentCount;
    /* free memory allocated for bit file */
    free(stream);

    return(fp);
}

/***************************************************************************
*   Function   : BitMemoryByteAlign
*   Description: This function aligns the bitfile to the nearest byte.  For
*                output files, this means writing out the bit buffer with
*                extra bits set to 0.  For input files, this means flushing
*                the bit buffer.
*   Parameters : stream - pointer to bit file stream to align
*   Effects    : Flushes out the bit buffer.
*   Returned   : EOF if stream is NULL or write fails.  Writes return the
*                byte aligned contents of the bit buffer.  Reads returns
*                the unaligned contents of the bit buffer.
***************************************************************************/
int BitMemoryByteAlign(bit_memory_t *stream)
{
    int returnValue;

    if (stream == NULL)
    {
        return(EOF);
    }

    returnValue = stream->bitBuffer;

    if ((stream->mode == BM_WRITE) || (stream->mode == BM_APPEND))
    {
        /* write out any unwritten bits */
        if (stream->bitCount != 0)
        {
            (stream->bitBuffer) <<= 8 - (stream->bitCount);
            if(stream->currentCount >= stream->size){
                return EOF;
            }
            stream->fp[stream->currentCount++] = stream->bitBuffer;
            
            // fputc(stream->bitBuffer, stream->fp);   /* handle error? */
        }
    }

    stream->bitBuffer = 0;
    stream->bitCount = 0;

    return (returnValue);
}

/***************************************************************************
*   Function   : BitMemoryFlushOutput
*   Description: This function flushes the output bit buffer.  This means
*                left justifying any pending bits, and filling spare bits
*                with the fill value.
*   Parameters : stream - pointer to bit file stream to align
*                onesFill - non-zero if spare bits are filled with ones
*   Effects    : Flushes out the bit buffer, filling spare bits with ones
*                or zeros.
*   Returned   : EOF if stream is NULL or not writeable.  Otherwise, the
*                bit buffer value written. -1 if no data was written.
***************************************************************************/
int BitMemoryFlushOutput(bit_memory_t *stream, const unsigned char onesFill)
{
    int returnValue;

    if (stream == NULL)
    {
        return(EOF);
    }

    returnValue = 0;

    /* write out any unwritten bits */
    if (stream->bitCount != 0)
    {
        stream->bitBuffer <<= (8 - stream->bitCount);

        if (onesFill)
        {
            stream->bitBuffer |= (0xFF >> stream->bitCount);
        }
        if(stream->currentCount >= stream->size){
            return EOF;
        }
        stream->fp[stream->currentCount++] = stream->bitBuffer;
    }

    stream->bitBuffer = 0;
    stream->bitCount = 0;

    return (returnValue);
}

/***************************************************************************
*   Function   : BitMemoryGetChar
*   Description: This function returns the next byte from the file passed as
*                a parameter.
*   Parameters : stream - pointer to bit file stream to read from
*   Effects    : Reads next byte from file and updates buffer accordingly.
*   Returned   : EOF if a whole byte cannot be obtained.  Otherwise,
*                the character read.
***************************************************************************/
int BitMemoryGetChar(bit_memory_t *stream)
{
    int returnValue;
    unsigned char tmp;

    if (stream == NULL)
    {
        return(EOF);
    }

    returnValue = stream->size > stream->currentCount ? stream->fp[stream->currentCount++] : EOF;// fgetc(stream->fp);
    

    if (stream->bitCount == 0)
    {
        /* we can just get byte from file */
        return returnValue;
    }

    /* we have some buffered bits to return too */
    if (returnValue != EOF)
    {
        /* figure out what to return */
        tmp = ((unsigned char)returnValue) >> (stream->bitCount);
        tmp |= ((stream->bitBuffer) << (8 - (stream->bitCount)));

        /* put remaining in buffer. count shouldn't change. */
        stream->bitBuffer = returnValue;

        returnValue = tmp;
    }

    return returnValue;
}

/***************************************************************************
*   Function   : BitMemoryPutChar
*   Description: This function writes the byte passed as a parameter to the
*                file passed a parameter.
*   Parameters : c - the character to be written
*                stream - pointer to bit file stream to write to
*   Effects    : Writes a byte to the file and updates buffer accordingly.
*   Returned   : On success, the character written, otherwise EOF.
***************************************************************************/
int BitMemoryPutChar(const int c, bit_memory_t *stream)
{
    unsigned char tmp;

    if (stream == NULL)
    {
        return(EOF);
    }

    if (stream->bitCount == 0)
    {
        /* we can just put byte from file */
        if(stream->currentCount >= stream->size){
            return EOF;
        }
        stream->fp[stream->currentCount++] = c;
        return 0;
        //return fputc(c, stream->fp);
    }

    /* figure out what to write */
    tmp = ((unsigned char)c) >> (stream->bitCount);
    tmp = tmp | ((stream->bitBuffer) << (8 - stream->bitCount));

    if(stream->currentCount >= stream->size){
        return EOF;
    }
    stream->fp[stream->currentCount++] = tmp;
    stream->bitBuffer = c;
    // if (fputc(tmp, stream->fp) != EOF)
    // {
    //     /* put remaining in buffer. count shouldn't change. */
    //     stream->bitBuffer = c;
    // }
    // else
    // {
    //     return EOF;
    // }

    return tmp;
}

/***************************************************************************
*   Function   : BitMemoryGetBit
*   Description: This function returns the next bit from the file passed as
*                a parameter.  The bit value returned is the msb in the
*                bit buffer.
*   Parameters : stream - pointer to bit file stream to read from
*   Effects    : Reads next bit from bit buffer.  If the buffer is empty,
*                a new byte will be read from the file.
*   Returned   : 0 if bit == 0, 1 if bit == 1, and EOF if operation fails.
***************************************************************************/
int BitMemoryGetBit(bit_memory_t *stream)
{
    int returnValue;

    if (stream == NULL)
    {
        return(EOF);
    }

    if (stream->bitCount == 0)
    {
        /* buffer is empty, read another character */
        returnValue = stream->size > stream->currentCount ? stream->fp[stream->currentCount++] : EOF;
        // if ((returnValue = fgetc(stream->fp)) == EOF)
        if ((returnValue) == EOF)
        {
            return EOF;
        }
        else
        {
            stream->bitCount = 8;
            stream->bitBuffer = returnValue;
        }
    }

    /* bit to return is msb in buffer */
    stream->bitCount--;
    returnValue = (stream->bitBuffer) >> (stream->bitCount);

    return (returnValue & 0x01);
}

/***************************************************************************
*   Function   : BitMemoryPutBit
*   Description: This function writes the bit passed as a parameter to the
*                file passed a parameter.
*   Parameters : c - the bit value to be written
*                stream - pointer to bit file stream to write to
*   Effects    : Writes a bit to the bit buffer.  If the buffer has a byte,
*                the buffer is written to the file and cleared.
*   Returned   : On success, the bit value written, otherwise EOF.
***************************************************************************/
int BitMemoryPutBit(const int c, bit_memory_t *stream)
{
    int returnValue = c;

    if (stream == NULL)
    {
        return(EOF);
    }

    stream->bitCount++;
    stream->bitBuffer <<= 1;

    if (c != 0)
    {
        stream->bitBuffer |= 1;
    }

    /* write bit buffer if we have 8 bits */
    if (stream->bitCount == 8)
    {
        if(stream->currentCount >= stream->size){
            return EOF;
        }
        stream->fp[stream->currentCount++] = stream->bitBuffer;
        // if (fputc(stream->bitBuffer, stream->fp) == EOF)
        // {
        //     returnValue = EOF;
        // }

        /* reset buffer */
        stream->bitCount = 0;
        stream->bitBuffer = 0;
    }

    return returnValue;
}

/***************************************************************************
*   Function   : BitMemoryGetBits
*   Description: This function reads the specified number of bits from the
*                file passed as a parameter and writes them to the
*                requested memory location (msb to lsb).
*   Parameters : stream - pointer to bit file stream to read from
*                bits - address to store bits read
*                count - number of bits to read
*   Effects    : Reads bits from the bit buffer and file stream.  The bit
*                buffer will be modified as necessary.
*   Returned   : EOF for failure, otherwise the number of bits read.  If
*                an EOF is reached before all the bits are read, bits
*                will contain every bit through the last complete byte.
***************************************************************************/
int BitMemoryGetBits(bit_memory_t *stream, void *bits, const unsigned int count)
{
    unsigned char *bytes, shifts;
    int offset, remaining, returnValue;

    bytes = (unsigned char *)bits;

    if ((stream == NULL) || (bits == NULL))
    {
        return(EOF);
    }

    offset = 0;
    remaining = count;

    /* read whole bytes */
    while (remaining >= 8)
    {
        returnValue = BitMemoryGetChar(stream);

        if (returnValue == EOF)
        {
            return EOF;
        }

        bytes[offset] = (unsigned char)returnValue;
        remaining -= 8;
        offset++;
    }

    if (remaining != 0)
    {
        /* read remaining bits */
        shifts = 8 - remaining;
        bytes[offset] = 0;

        while (remaining > 0)
        {
            returnValue = BitMemoryGetBit(stream);

            if (returnValue == EOF)
            {
                return EOF;
            }

            bytes[offset] <<= 1;
            bytes[offset] |= (returnValue & 0x01);
            remaining--;
        }

        /* shift last bits into position */
        bytes[offset] <<= shifts;
    }

    return count;
}

/***************************************************************************
*   Function   : BitMemoryPutBits
*   Description: This function writes the specified number of bits from the
*                memory location passed as a parameter to the file passed
*                as a parameter.   Bits are written msb to lsb.
*   Parameters : stream - pointer to bit file stream to write to
*                bits - pointer to bits to write
*                count - number of bits to write
*   Effects    : Writes bits to the bit buffer and file stream.  The bit
*                buffer will be modified as necessary.
*   Returned   : EOF for failure, otherwise the number of bits written.  If
*                an error occurs after a partial write, the partially
*                written bits will not be unwritten.
***************************************************************************/
int BitMemoryPutBits(bit_memory_t *stream, void *bits, const unsigned int count)
{
    unsigned char *bytes, tmp;
    int offset, remaining, returnValue;

    bytes = (unsigned char *)bits;

    if ((stream == NULL) || (bits == NULL))
    {
        return(EOF);
    }

    offset = 0;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
        returnValue = BitMemoryPutChar(bytes[offset], stream);

        if (returnValue == EOF)
        {
            return EOF;
        }

        remaining -= 8;
        offset++;
    }

    if (remaining != 0)
    {
        /* write remaining bits */
        tmp = bytes[offset];

        while (remaining > 0)
        {
            returnValue = BitMemoryPutBit((tmp & 0x80), stream);

            if (returnValue == EOF)
            {
                return EOF;
            }

            tmp <<= 1;
            remaining--;
        }
    }

    return count;
}

/***************************************************************************
*   Function   : BitMemoryGetBitsNum
*   Description: This function provides a machine independent layer that
*                allows a single function call to stuff an arbitrary number
*                of bits into an integer type variable.
*   Parameters : stream - pointer to bit file stream to read from
*                bits - address to store bits read
*                count - number of bits to read
*                size - sizeof type containing "bits"
*   Effects    : Calls a function that reads bits from the bit buffer and
*                file stream.  The bit buffer will be modified as necessary.
*                the bits will be written to "bits" from least significant
*                byte to most significant byte.
*   Returned   : EOF for failure, -ENOTSUP unsupported architecture,
*                otherwise the number of bits read by the called function.
***************************************************************************/
int BitMemoryGetBitsNum(bit_memory_t *stream, void *bits, const unsigned int count,
    const size_t size)
{
    if ((stream == NULL) || (bits == NULL))
    {
        return EOF;
    }

    if (NULL == stream->GetBitsNumFunc)
    {
        return -ENOTSUP;
    }

    /* call function that correctly handles endianess */
    return (stream->GetBitsNumFunc)(stream, bits, count, size);
}

/***************************************************************************
*   Function   : BitMemoryGetBitsLE   (Little Endian)
*   Description: This function reads the specified number of bits from the
*                file passed as a parameter and writes them to the
*                requested memory location (LSB to MSB).
*   Parameters : stream - pointer to bit file stream to read from
*                bits - address to store bits read
*                count - number of bits to read
*                size - sizeof type containing "bits"
*   Effects    : Reads bits from the bit buffer and file stream.  The bit
*                buffer will be modified as necessary.  bits is treated as
*                a little endian integer of length >= (count/8) + 1.
*   Returned   : EOF for failure, otherwise the number of bits read.  If
*                an EOF is reached before all the bits are read, bits
*                will contain every bit through the last successful read.
***************************************************************************/
static int BitMemoryGetBitsLE(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size)
{
    unsigned char *bytes;
    int offset, remaining, returnValue;

    (void)size;
    bytes = (unsigned char *)bits;
    offset = 0;
    remaining = count;

    /* read whole bytes */
    while (remaining >= 8)
    {
        returnValue = BitMemoryGetChar(stream);

        if (returnValue == EOF)
        {
            return EOF;
        }

        bytes[offset] = (unsigned char)returnValue;
        remaining -= 8;
        offset++;
    }

    if (remaining != 0)
    {
        /* read remaining bits */
        while (remaining > 0)
        {
            returnValue = BitMemoryGetBit(stream);

            if (returnValue == EOF)
            {
                return EOF;
            }

            bytes[offset] <<= 1;
            bytes[offset] |= (returnValue & 0x01);
            remaining--;
        }

    }

    return count;
}

/***************************************************************************
*   Function   : BitMemoryGetBitsBE   (Big Endian)
*   Description: This function reads the specified number of bits from the
*                file passed as a parameter and writes them to the
*                requested memory location (LSB to MSB).
*   Parameters : stream - pointer to bit file stream to read from
*                bits - address to store bits read
*                count - number of bits to read
*                size - sizeof type containing "bits"
*   Effects    : Reads bits from the bit buffer and file stream.  The bit
*                buffer will be modified as necessary.  bits is treated as
*                a big endian integer of length size.
*   Returned   : EOF for failure, otherwise the number of bits read.  If
*                an EOF is reached before all the bits are read, bits
*                will contain every bit through the last successful read.
***************************************************************************/
static int BitMemoryGetBitsBE(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size)
{
    unsigned char *bytes;
    int offset, remaining, returnValue;

    if (count > (size * 8))
    {
        /* too many bits to read */
        return EOF;
    }

    bytes = (unsigned char *)bits;

    offset = size - 1;
    remaining = count;

    /* read whole bytes */
    while (remaining >= 8)
    {
        returnValue = BitMemoryGetChar(stream);

        if (returnValue == EOF)
        {
            return EOF;
        }

        bytes[offset] = (unsigned char)returnValue;
        remaining -= 8;
        offset--;
    }

    if (remaining != 0)
    {
        /* read remaining bits */
        while (remaining > 0)
        {
            returnValue = BitMemoryGetBit(stream);

            if (returnValue == EOF)
            {
                return EOF;
            }

            bytes[offset] <<= 1;
            bytes[offset] |= (returnValue & 0x01);
            remaining--;
        }

    }

    return count;
}

/***************************************************************************
*   Function   : BitMemoryPutBitsNum
*   Description: This function provides a machine independent layer that
*                allows a single function call to write an arbitrary number
*                of bits from an integer type variable into a file.
*   Parameters : stream - pointer to bit file stream to write to
*                bits - pointer to bits to write
*                count - number of bits to write
*                size - sizeof type containing "bits"
*   Effects    : Calls a function that writes bits to the bit buffer and
*                file stream.  The bit buffer will be modified as necessary.
*                the bits will be written to the file stream from least
*                significant byte to most significant byte.
*   Returned   : EOF for failure, ENOTSUP unsupported architecture,
*                otherwise the number of bits written.  If an error occurs
*                after a partial write, the partially written bits will not
*                be unwritten.
***************************************************************************/
int BitMemoryPutBitsNum(bit_memory_t *stream, void *bits, const unsigned int count,
    const size_t size)
{
    if ((stream == NULL) || (bits == NULL))
    {
        return EOF;
    }

    if (NULL == stream->PutBitsNumFunc)
    {
        return ENOTSUP;
    }

    /* call function that correctly handles endianess */
    return (stream->PutBitsNumFunc)(stream, bits, count, size);
}

/***************************************************************************
*   Function   : BitMemoryPutBitsLE   (Little Endian)
*   Description: This function writes the specified number of bits from the
*                memory location passed as a parameter to the file passed
*                as a parameter.   Bits are written LSB to MSB.
*   Parameters : stream - pointer to bit file stream to write to
*                bits - pointer to bits to write
*                count - number of bits to write
*                size - sizeof type containing "bits"
*   Effects    : Writes bits to the bit buffer and file stream.  The bit
*                buffer will be modified as necessary.  bits is treated as
*                a little endian integer of length >= (count/8) + 1.
*   Returned   : EOF for failure, otherwise the number of bits written.  If
*                an error occurs after a partial write, the partially
*                written bits will not be unwritten.
***************************************************************************/
static int BitMemoryPutBitsLE(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size)
{
    unsigned char *bytes, tmp;
    int offset, remaining, returnValue;

    (void)size;
    bytes = (unsigned char *)bits;
    offset = 0;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
        returnValue = BitMemoryPutChar(bytes[offset], stream);

        if (returnValue == EOF)
        {
            return EOF;
        }

        remaining -= 8;
        offset++;
    }

    if (remaining != 0)
    {
        /* write remaining bits */
        tmp = bytes[offset];
        tmp <<= (8 - remaining);

        while (remaining > 0)
        {
            returnValue = BitMemoryPutBit((tmp & 0x80), stream);

            if (returnValue == EOF)
            {
                return EOF;
            }

            tmp <<= 1;
            remaining--;
        }
    }

    return count;
}

/***************************************************************************
*   Function   : BitMemoryPutBitsBE   (Big Endian)
*   Description: This function writes the specified number of bits from the
*                memory location passed as a parameter to the file passed
*                as a parameter.   Bits are written LSB to MSB.
*   Parameters : stream - pointer to bit file stream to write to
*                bits - pointer to bits to write
*                count - number of bits to write
*   Effects    : Writes bits to the bit buffer and file stream.  The bit
*                buffer will be modified as necessary.  bits is treated as
*                a big endian integer of length size.
*   Returned   : EOF for failure, otherwise the number of bits written.  If
*                an error occurs after a partial write, the partially
*                written bits will not be unwritten.
***************************************************************************/
static int BitMemoryPutBitsBE(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size)
{
    unsigned char *bytes, tmp;
    int offset, remaining, returnValue;

    if (count > (size * 8))
    {
        /* too many bits to write */
        return EOF;
    }

    bytes = (unsigned char *)bits;
    offset = size - 1;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
        returnValue = BitMemoryPutChar(bytes[offset], stream);

        if (returnValue == EOF)
        {
            return EOF;
        }

        remaining -= 8;
        offset--;
    }

    if (remaining != 0)
    {
        /* write remaining bits */
        tmp = bytes[offset];
        tmp <<= (8 - remaining);

        while (remaining > 0)
        {
            returnValue = BitMemoryPutBit((tmp & 0x80), stream);

            if (returnValue == EOF)
            {
                return EOF;
            }

            tmp <<= 1;
            remaining--;
        }
    }

    return count;
}

/***************************************************************************
*   Function   : BitMemoryNotSupported
*   Description: This function returns -ENOTSUP.  It is called when a
*                Get/PutBits function is called on an unsupported
*                architecture.
*   Parameters : stream - not used
*                bits - not used
*                count - not used
*   Effects    : None
*   Returned   : -ENOTSUP
***************************************************************************/
static int BitMemoryNotSupported(bit_memory_t *stream, void *bits,
    const unsigned int count, const size_t size)
{
    (void)stream;
    (void)bits;
    (void)count;
    (void)size;

    return -ENOTSUP;
}
