-------lzss_spar.cpp: In function ‘int EncodeLZSSCpuGpu(FileStream*, FileStream*, ExecutionPlan, AppStatistics*, int)’:
-------lzss_spar.cpp:122:3: warning: attributes at the beginning of statement are ignored [-Wattributes]
-------   [[spar::Stage,spar::Input(matcher,buffer,i,fpOut,matches_offset,matches_length,matchSize,metrics,currentMatchCount,current,isLast),spar::Output(current,buffer,matches_offset,matches_length,matchSize,fpOut), spar::Replicate(workers)]]
-------   ^
-------lzss_spar.cpp:131:3: warning: attributes at the beginning of statement are ignored [-Wattributes]
-------   [[spar::Stage, spar::Input(current,buffer,matches_offset,matches_length,matchSize,fpOut)]]
-------   ^
-------lzss_spar.cpp:84:2: warning: attributes at the beginning of statement are ignored [-Wattributes]
-------  [[spar::ToStream,spar::Input(fpIn,fpOut,lastBuffer,nextJump,current,batchSizeWithWindow,batchSize,matcher,isLast,metrics,currentMatchCount)]]
-------  ^

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
#include <ff/pipeline.hpp>
 
#include <ff/farm.hpp>
 
using namespace ff; 
namespace spar{
	static inline ssize_t get_mac_core() {
		ssize_t n = 1; 
		FILE * f; 
		f = popen("cat /proc/cpuinfo |grep processor | wc -l","r"); 
		if(fscanf(f,"%ld",& n) == EOF)
		{
			pclose (f); 
			return n;
		} 
		pclose (f); 
		return n;
	} 
	static inline ssize_t get_env_num_workers() {
		ssize_t n = 1; 
		FILE * f; 
		f = popen("echo $SPAR_NUM_WORKERS","r"); 
		if(fscanf(f,"%ld",& n) == EOF)
		{
			pclose (f); 
			return n;
		} 
		pclose (f); 
		return n;
	} 
	static inline ssize_t get_Num_Workers() {
		ssize_t w_size = get_env_num_workers(); 
		if(w_size > 0)
		{
			return w_size;
		} 
		return get_mac_core();
	}
} 
struct struct_spar0{
	struct_spar0(MatcherBase * matcher,char * buffer,int i,FileStream * fpOut,int * matches_offset,int * matches_length,int matchSize,AppStatistics * metrics,int currentMatchCount,int current,bool isLast) : matcher(matcher),buffer(buffer),i(i),fpOut(fpOut),matches_offset(matches_offset),matches_length(matches_length),matchSize(matchSize),metrics(metrics),currentMatchCount(currentMatchCount),current(current),isLast(isLast) {
	} 
	; 
	MatcherBase * matcher; 
	char * buffer; 
	int i; 
	FileStream * fpOut; 
	int * matches_offset; 
	int * matches_length; 
	int matchSize; 
	AppStatistics * metrics; 
	int currentMatchCount; 
	int current; 
	bool isLast;
}; 
struct_spar0 * Stage_spar00(struct_spar0 * Input_spar,ff_node *const) {
	{
		printf("Current Stage:%i\n",Input_spar -> current); 
		Input_spar -> metrics -> StartFindMatch(); 
		Input_spar -> matcher -> FindMatchBatch(Input_spar -> buffer,Input_spar -> i,Input_spar -> matches_length,Input_spar -> matches_offset,& Input_spar -> matchSize,Input_spar -> isLast,Input_spar -> currentMatchCount); 
		Input_spar -> metrics -> StopFindMatch(); 
		Input_spar -> currentMatchCount += Input_spar -> i-MAX_CODED;
	} 
	return Input_spar;
} 
struct_spar0 * Stage_spar01(struct_spar0 * Input_spar,ff_node *const) {
	{
		int x; 
		
		for(x = nextJump; x < Input_spar -> matchSize;x++)
		{
			#ifdef PRINTOFFSETS
 
			printf("%i %i\n",Input_spar -> matches_offset[x],Input_spar -> matches_length[x]); 
			#endif
 
			if(Input_spar -> matches_length[x] <= MAX_UNCODED)
			{
				Input_spar -> fpOut -> PutBit(UNCODED); 
				Input_spar -> fpOut -> PutChar(Input_spar -> buffer[x+WINDOW_SIZE]);
			} else 
			{
				unsigned int adjustedLen; 
				adjustedLen = Input_spar -> matches_length[x]-(MAX_UNCODED+1); 
				Input_spar -> fpOut -> PutBit(ENCODED); 
				Input_spar -> fpOut -> PutBitsNum(Input_spar -> matches_offset+x,OFFSET_BITS,sizeof(unsigned int)); 
				Input_spar -> fpOut -> PutBitsNum(& adjustedLen,LENGTH_BITS,sizeof(unsigned int));
			} 
			if(Input_spar -> matches_length[x] > 2)
			{
				x += Input_spar -> matches_length[x]-1;
			}
		} 
		delete[]Input_spar -> matches_offset; 
		delete[]Input_spar -> matches_length; 
		delete[]Input_spar -> buffer; 
		nextJump = x-Input_spar -> matchSize;
	} 
	delete Input_spar; 
	return (struct_spar0 *)GO_ON;
} 
struct ToStream_spar0 : ff_node_t < struct_spar0 >{
	FileStream * fpIn; 
	FileStream * fpOut; 
	char * lastBuffer; 
	int nextJump; 
	int current; 
	int batchSizeWithWindow; 
	int batchSize; 
	MatcherBase * matcher; 
	bool isLast; 
	AppStatistics * metrics; 
	int currentMatchCount; 
	struct_spar0 * svc(struct_spar0 * Input_spar) {
		
		while(true)
		{
			int c = ' '; 
			if(isLast)
			{
				break;
			} 
			char * buffer = new char [batchSizeWithWindow]; 
			if(current == 0)
			{
				memset(buffer,' ',(WINDOW_SIZE)* sizeof(char));
			} else 
			{
				memcpy(buffer,lastBuffer,sizeof(char)*(WINDOW_SIZE+MAX_CODED));
			} 
			int i; 
			
			for(i = WINDOW_SIZE+(current == 0 ? 0 : MAX_CODED); i < batchSizeWithWindow && (c = fpIn -> GetChar()) != EOF;i++)
			{
				buffer[i] = c;
			} 
			isLast = i != batchSizeWithWindow; 
			if(isLast)
			{
				buffer[i] = '\0'; 
				memset(buffer+i,' ',(batchSizeWithWindow-i)*sizeof(char));
			} 
			if(! isLast)
			{
				memcpy(lastBuffer,buffer+(batchSizeWithWindow-WINDOW_SIZE-MAX_CODED),sizeof(char)*(MAX_CODED+WINDOW_SIZE));
			} 
			current++; 
			int matchSize = 0; 
			int * matches_offset = new int [batchSize]; 
			int * matches_length = new int [batchSize]; 
			struct_spar0 * stream_spar = new struct_spar0 (matcher,buffer,i,fpOut,matches_offset,matches_length,matchSize,metrics,currentMatchCount,current,isLast); 
			ff_send_out (stream_spar); 
			;
		} 
		return EOS;
	}
}; 
int EncodeLZSSCpuGpu(FileStream * fpIn,FileStream * fpOut,ExecutionPlan plan,AppStatistics * metrics,int workers) {
	ToStream_spar0 ToStream_spar0_call; 
	ff_OFarm < struct_spar0 > Stage_spar00_call(Stage_spar00,workers); 
	Stage_spar00_call.setEmitterF(ToStream_spar0_call); 
	ff_node_F < struct_spar0 > Stage_spar01_call (Stage_spar01); 
	Stage_spar00_call.setCollectorF(Stage_spar01_call); 
	MatcherBase * matcher; 
	switch(plan)
	{
		case GPU_CUDA : 
		matcher = (new MatcherCuda); 
		break; 
		case GPU_OPENACC : 
		matcher = (new MatcherOpenAcc); 
		break; 
		case GPU_OPENCL : 
		matcher = (new MatcherOpenCl); 
		break; 
		case CPU_SEQUENTIAL : 
		default : 
		matcher = (new MatcherSequential); 
		break;
	} 
	matcher -> Init(); 
	int batchSize = BATCH_SIZE; 
	int batchSizeWithWindow = batchSize+WINDOW_SIZE+MAX_CODED; 
	int current = 0; 
	nextJump = 0; 
	int currentMatchCount = 0; 
	char * lastBuffer = new char [WINDOW_SIZE+MAX_CODED]; 
	isLast = false; 
	ToStream_spar0_call.fpIn = fpIn; 
	ToStream_spar0_call.fpOut = fpOut; 
	ToStream_spar0_call.lastBuffer = lastBuffer; 
	ToStream_spar0_call.nextJump = nextJump; 
	ToStream_spar0_call.current = current; 
	ToStream_spar0_call.batchSizeWithWindow = batchSizeWithWindow; 
	ToStream_spar0_call.batchSize = batchSize; 
	ToStream_spar0_call.matcher = matcher; 
	ToStream_spar0_call.isLast = isLast; 
	ToStream_spar0_call.metrics = metrics; 
	ToStream_spar0_call.currentMatchCount = currentMatchCount; 
	if(Stage_spar00_call.run_and_wait_end() < 0)
	{
		error("Running farm\n"); 
		exit(1);
	} 
	fpOut -> Flush(); 
	delete[]lastBuffer; 
	#define MICRO_TO_SECONDS(v) v/1000
 
	std::cout<<"timeSpentOnMemoryHostToDevice: "<<MICRO_TO_SECONDS(matcher -> timeSpentOnMemoryHostToDevice)<<" seconds"<<std::endl; 
	std::cout<<"timeSpentOnMemoryDeviceToHost: "<<MICRO_TO_SECONDS(matcher -> timeSpentOnMemoryDeviceToHost)<<" seconds"<<std::endl; 
	std::cout<<"timeSpentOnKernel: "<<MICRO_TO_SECONDS(matcher -> timeSpentOnKernel)<<" seconds"<<std::endl; 
	#undef MICRO_TO_SECONDS
 
	delete matcher; 
	return 0;
}
