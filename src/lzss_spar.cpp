/***************************************************************************
*                 Lempel, Ziv, Storer, and Szymanski Encoding
*
*   File    : lzss.c
*   Purpose : Use lzss coding (Storer and Szymanski's modified LZ77) to
*             compress lzss data files.
*   Author  : Michael Dipperstein
*   Date    : November 28, 2014
*
****************************************************************************
*
* LZss: An ANSI C LZSS Encoding/Decoding Routines
* Copyright (C) 2003 - 2007, 2014 by
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
#include <string.h>
#include <memory>
#include <iostream>
#include <errno.h>
#include "lzlocal.h"
#include "lzss.h"
#include "bitfile.h"
#include "statistics.h"
#include "matchers/matcher_base.h"
#include "file_stream.h"
#include <algorithm>

int nextJump = 0;
bool isLast = false;
int EncodeLZSSCpuGpu(FileStream* fpIn, FileStream *fpOut,ExecutionPlan plan, AppStatistics* metrics, int workers) {
	
	/* convert output file to bitfile */
	//Smart pointer to Matcher
	MatcherBase * matcher;
	switch(plan){
		case GPU_CUDA:
			matcher = (new MatcherCuda);
			break;
		case GPU_OPENACC:
			matcher = (new MatcherOpenAcc);
			break;

		case GPU_OPENCL:
			matcher = (new MatcherOpenCl);
			break;
		case CPU_SEQUENTIAL:
		default:
			matcher = (new MatcherSequential);
			break;
	}
	matcher->Init();
	int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	//Read the batch size
	
	//encoded_string_t matches[LzssBatchSize];
	
	int current = 0;
	nextJump = 0;
	int currentMatchCount = 0;
	char* lastBuffer = new char[WINDOW_SIZE + MAX_CODED];
	isLast = false;
	[[spar::ToStream,spar::Input(fpIn,fpOut,lastBuffer,nextJump,current,batchSizeWithWindow,matcher,isLast,metrics,currentMatchCount)]]
	while (true) {
		int c = ' ';
		
		if (isLast) {
			break;
		}
		char* buffer = new char[batchSizeWithWindow ];
		if (current == 0) {
			memset(buffer, ' ', (WINDOW_SIZE) * sizeof(char));
		}
		else {
			memcpy(buffer,lastBuffer,sizeof(char) * (WINDOW_SIZE + MAX_CODED) );
		}
		int i;
		for (i = WINDOW_SIZE + (current == 0 ? 0 : MAX_CODED ); i < batchSizeWithWindow && (c = fpIn->GetChar()) != EOF; i++)
		{
			buffer[i] = c;
		}
		isLast = i != batchSizeWithWindow;
		if (isLast) {
			buffer[i] = '\0';
			memset(buffer + i, ' ', (batchSizeWithWindow - i) * sizeof(char));
			
		}
		
		if(!isLast) {
			memcpy(lastBuffer,buffer +(batchSizeWithWindow - WINDOW_SIZE - MAX_CODED),sizeof(char) * (MAX_CODED + WINDOW_SIZE) );
		}
	
		current++;
		
		
		int matchSize = 0;
		//Find matches in batch
		int* matches_offset = new int[LzssBatchSize];
		int* matches_length = new int[LzssBatchSize];
	
		[[spar::Stage,spar::Input(matcher,buffer,i,fpOut,matches_offset,matches_length,matchSize,metrics,currentMatchCount,current,isLast),spar::Output(current,buffer,matches_offset,matches_length,matchSize,fpOut), spar::Replicate(workers)]]
		{
		//printf("Current Stage:%i\n", current);
		metrics->StartFindMatch();
		matcher->FindMatchBatch(buffer, i, matches_length,matches_offset, &matchSize, isLast, currentMatchCount, current);
		metrics->StopFindMatch();
		currentMatchCount += i - MAX_CODED;
		
		}
		[[spar::Stage, spar::Input(current,buffer,matches_offset,matches_length,matchSize,fpOut)]]
		{
			
			//printf("Current Save:%i\n", current);
		int x;
		for (x = nextJump; x < matchSize; x++) {
			
			#ifdef PRINTOFFSETS
			printf("%i %i\n", matches_offset[x], matches_length[x]);
			#endif
			//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				/* not long enough match.  write uncoded flag and character */
				fpOut->PutBit(UNCODED);
				fpOut->PutChar(buffer[x + WINDOW_SIZE]);
			}
			else
			{
				unsigned int adjustedLen;

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				fpOut->PutBit(ENCODED);
				fpOut->PutBitsNum(matches_offset +x, OFFSET_BITS,
					sizeof(unsigned int));
				fpOut->PutBitsNum( &adjustedLen, LENGTH_BITS,
					sizeof(unsigned int));
			}
			// #endif
			if (matches_length[x] > 2) {
				x += matches_length[x] - 1;
			}
		}
		delete[] matches_offset;
		delete[] matches_length;
		delete[] buffer;

		nextJump = x - matchSize;
		}
		
	}

	// BitFileToFILE(bfpOut);
	fpOut->Flush();
	
	delete[] lastBuffer;
        #define MICRO_TO_SECONDS(v) v/1000

	std::cout
		<< "timeSpentOnMemoryHostToDevice: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnMemoryHostToDevice)
		<< " seconds"
		<< std::endl;
	std::cout
		<< "timeSpentOnMemoryDeviceToHost: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnMemoryDeviceToHost)
		<< " seconds"
		<< std::endl;
	std::cout
		<< "timeSpentOnKernel: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnKernel)
		<< " seconds"
		<< std::endl;
		#undef MICRO_TO_SECONDS
	
	delete matcher;
	return 0;
}
