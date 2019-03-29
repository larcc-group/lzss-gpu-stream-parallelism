LZSS GPU is a Lempel-Ziv-Storer-Szymansk compression algorithm optimized for GPUs.

# How to Cite LZSS

Stein, C.; Griebler, D.; Danelutto, M.; Fernandes, L. G. **Stream Parallelism on the LZSS Data Compression Application for Multi-Cores with GPUs**. *27th Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP)*, pp. 247-251, IEEE, Pavia, Italy, 2019. [[PDF](https://dx.doi.org/10.1109/EMPDP.2019.8671624)]

```bibtex
@inproceedings{larcc:stream_parallelism_lzss_gpu:PDP:19,
    author={Charles Michael Stein and Dalvan Griebler and Marco Danelutto and Luiz Gustavo Fernandes},
    title={{Stream Parallelism on the LZSS Data Compression Application for Multi-Cores with GPUs}},
    booktitle={27th Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP)},
    pages={247-251},
    publisher={IEEE},
    address={Pavia, Italy},
    month={February},
    year={2019},
    doi={10.1109/EMPDP.2019.8671624},
    url={https://ieeexplore.ieee.org/document/8671624},
}

```

# lzss-gpu-stream-parallelism
This repository has the parallel and sequential implementations for the Lempel-Ziv-Storer-Szymanski (LZSS) data compression applications. We introduced stream parallelism for CPU (using SPar) and GPU (using CUDA and OpenCL).

## Compiling
To compile the project, you need to have the following prerequisites installed.
 - CUDA
 - SPar in your home (https://github.com/dalvangriebler/SPar)

Once you have all the prerequisites installed, run the following command inside *src* folder:

```make```

This will generate the *app* runnable file. PS: You wont be able to run it on on Windows, because SPar at the time we are publishing it can only be ran on linux.
## Using Lzss on GPU

### Compression
```
    ./src/app -c -i <input_file_name> -o <output_file_name>
```
By default, this command will use sequential version of compression, you can change the mode to use GPU by passing the *-p* argument
```
    ./src/app -c -i <input_file_name> -o <output_file_name> -p <cuda|opencl|cpu_original|cpu_sequential>
```
To use multiple GPU, pass in the argument *-g*, indicating the GPU ids separated by comma
```
    ./src/app -c -i <input_file_name> -o <output_file_name> -p cuda -g 0,1
```
For benchmarking purposes, you can pass the *-m* argument to run the test in memory
### Decompression
The decompression is only available on a serial implementation, thus you only need to provide the input file  path and the output path by using this command:
```
    ./src/app -d -i <input_file_name> -o <output_file_name>
```
### Help
```
    ./src/app --help
```
## Downloading datasets
The datasets used in our work can be downloaded by executing this command:
```
cd data
./download_dataset.sh
```
