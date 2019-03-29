
#define OFFSET_BITS     12
#define LENGTH_BITS     4

#if (((1 << (OFFSET_BITS + LENGTH_BITS)) - 1) > UINT_MAX)
#error "Size of encoded data must not exceed the size of an unsigned int"
#endif

/* We want a sliding window*/
#define WINDOW_SIZE     (1 << OFFSET_BITS)

/* maximum match length not encoded and maximum length encoded (4 bits) */
#define MAX_UNCODED     2
#define MAX_CODED       ((1 << LENGTH_BITS) + MAX_UNCODED)

#define ENCODED     0       /* encoded string */
#define UNCODED     1       /* unencoded character */

#define Wrap(value, limit) \
    (((value) < (limit)) ? (value) : ((value) - (limit)))

__kernel
void  FindMatchBatchKernel(__global char* buffer, int bufferSize,__global int* matches_length, __global int* matches_offset,int bufferSizeAdjusted, int currentMatchCount,  int isLast) {
    
    int idX = get_global_id(0);//blockIdx.x*blockDim.x+threadIdx.x;
    int i = WINDOW_SIZE + idX;
    int beginSearch = idX;
    if( i >= bufferSizeAdjusted){
        return;
    }

    //Uncoded Lookahead optimization
    char current[MAX_CODED];
    for (int j = 0; j < MAX_CODED; j++)
    {
        current[j] = buffer[i + j];
    }
    
    int length = 0;
    int offset = 0;
    int windowHead = (currentMatchCount + idX) % WINDOW_SIZE;
  
    int currentOffset = 0;

    // char* current = buffer;
    int j = 0;
    while (1) {
        if (current[0] == buffer[beginSearch  + Wrap((currentOffset), WINDOW_SIZE)]) {
            /* we matched one. how many more match? */
            j = 1;
            
            while (
                current[j] == buffer[beginSearch  + Wrap((currentOffset + j),WINDOW_SIZE)]
                && (!isLast ||
                ( beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < bufferSizeAdjusted
                && i + j < bufferSizeAdjusted) )
                ) {
                        
                if (j >= MAX_CODED) {
                    break;
                }					
                j++;
            }
            
            if (j > length) {
                
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
        
        if (currentOffset == WINDOW_SIZE) {
            break;
        }

    }
    matches_offset[idX] = offset;
    matches_length[idX] = length;
}




__kernel void FindMatchBatchKernelWithoutBuffer(__global unsigned char *buffer, int bufferSize,__global int *matches_length,__global int *matches_offset)
{
    int idX = get_global_id(0);//blockIdx.x*blockDim.x+threadIdx.x;

    int i = idX;
    int beginSearch = idX - WINDOW_SIZE;
    if (i >= bufferSize)
    {
        return;
    }

    int length = 0;
    int offset = 0;
    int windowHead = ( idX) % WINDOW_SIZE;

    int currentOffset = 0;

    //Uncoded Lookahead optimization
    char current[MAX_CODED];
    //for (int j = 0; j < MAX_CODED && i + j < bufferSizeAdjusted; j++)
    for (int j = 0; j < MAX_CODED; j++)
    {
        current[j] = buffer[i + j];
    }

    //First WINDOW_SIZE bits will always be ' ', optimize begging where data really is
    if(beginSearch < -MAX_CODED){
        currentOffset = (beginSearch * -1) - MAX_CODED;
    }
//    char* current = buffer + i;
    int j = 0;
    while (1)
    {
        if (current[0] == (beginSearch + Wrap((currentOffset), WINDOW_SIZE) < 0? ' ': buffer[beginSearch + Wrap((currentOffset), WINDOW_SIZE)]))
        {
            /* we matched one. how many more match? */
            j = 1;

            while (
              current[j] == (beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < 0?' ':buffer[beginSearch + Wrap((currentOffset + j), WINDOW_SIZE)]) &&  
                beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < bufferSize && i + j < bufferSize)
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