#include "matcher_base.h"
#include "lzlocal.h"
#include <iostream>

int MatcherSequential::Init()
{
	MatcherBase::Init();
	return 0;
}
int MatcherSequential::FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch)
{

	int bufferSizeAdjusted = bufferSize - MAX_CODED;
	if (isLast)
	{
		bufferSizeAdjusted += MAX_CODED;
	}
	int matchCount = bufferSizeAdjusted - WINDOW_SIZE;
	*matchSize = matchCount;
	for (int idX = 0; idX < matchCount; idX++)
	{
		int i = WINDOW_SIZE + idX;
		int beginSearch = idX;
		

		int length = 0;
		int offset = 0;
		int windowHead = (currentMatchCount + idX) % WINDOW_SIZE;

		int currentOffset = 0;

		// char* current = buffer;
		int j = 0;
		while (1)
		{
			if (buffer[i + 0] == buffer[beginSearch + Wrap((currentOffset), WINDOW_SIZE)])
			{
				/* we matched one. how many more match? */
				j = 1;

				while (
					buffer[i + j] == buffer[beginSearch + Wrap((currentOffset + j), WINDOW_SIZE)] && (!isLast ||
																									  (beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < bufferSizeAdjusted && i + j < bufferSizeAdjusted)))
				{

					if (j >= MAX_CODED)
					{
						break;
					}
					j++;
				}

				if (j > length)
				{

					length = j;
					offset = Wrap((currentOffset + windowHead), WINDOW_SIZE);
				}
			}

			if (j >= MAX_CODED)
			{
				length = MAX_CODED;
				break;
			}

			currentOffset++;

			if (currentOffset == WINDOW_SIZE)
			{
				break;
			}
		}
		matches_offset[idX] = offset;
		matches_length[idX] = length;
	}

	return 0;
}
