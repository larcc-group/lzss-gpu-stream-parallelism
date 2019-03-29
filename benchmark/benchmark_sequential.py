#!/usr/bin/python3
from __future__ import print_function
import os
import glob
import re
from time import gmtime, strftime,sleep
from subprocess import Popen, PIPE
import sqlite3
from sys import argv
from datetime import datetime

GPUS = [0,1,2,3]
PLANS = ["cpu_sequential_original"]
APP_FILE = "app"
dir_name = os.path.dirname(__file__)
EXE_PATH =os.path.abspath(os.path.join(dir_name, "../src"))
THREADS_CPU = [10,20,30,40,50,60,70,80]

gpus_arg = "" #  ",".join([str(x) for x in GPUS])
print(gpus_arg)
def log(text):
    print(text)
    with open("benchmark.log","a+") as log_file:
        log_file.write("{0}: {1}\n".format(datetime.now(),text))
    

class ItemResult:
    def __init__(self):
        self.mode = ""
        self.dataset = ""
        self.plan = ""
        self.file_statistics = "file"
        self.seq = 0
        self.input_file_size = 0.0
        self.output_file_size  = 0.0
        self.compress_ratio = 0.0
        self.input_processing = 0.0
        self.output_processing = 0.0
        self.gpu_host_to_device = 0.0
        self.gpu_device_to_host = 0.0
        self.gpu_kernel = 0.0
        self.total_time = 0.0
        self.finding_match = 0.0
        self.threads = 0


def main(args):
    if len(args) < 3:
        print("Usage: {0} <dataset> <repetitions> ".format(args[0]))
        return

    log("Dataset {0}".format(args[1]))
    log("Repetitions {0}".format(int(args[2])))
    run_for_benchmark("benchmark.db",args[1],int(args[2]))

def run_for_benchmark( database_name,dataset, repetitions):
    db = sqlite3.connect(database_name)
    # Get a cursor object
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS result(
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            dataset TEXT,
            mode TEXT,
            plan TEXT,
            seq int,
            threads int,
            input_file_size decimal(10,2),
            output_file_size  decimal(10,2),
            compress_ratio decimal(10,2),
            input_processing decimal(10,2),
            output_processing decimal(10,2),
            total_time decimal(10,6),
            gpu_host_to_device decimal(10,6),
            gpu_device_to_host decimal(10,6),
            gpu_kernel decimal(10,4),
            finding_match decimal(10,6),
            current_date TEXT,
            current_time TEXT
            )
    ''')
    db.commit()
    
    total = repetitions * len(PLANS)
    current = 0
    for i in range(repetitions):
        for plan in PLANS:
            break_threads = False
            for threads in range(len(THREADS_CPU)):
                if break_threads:
                    break_threads = False
                    break
            #try again 3 times
                for retrial in range(3):
                    try:
                        if threads == 1 and plan !="cpu_sequential" and plan !="cuda" and plan != "opencl":
                            break_threads = True
                            break

                        if plan =="cuda" or plan == "opencl":
                            if len(GPUS) == threads:
                                break_threads = True
                                break
                            gpus_arg =   ",".join([str(GPUS[i]) for i in range(0,threads + 1)])

                        if plan == "cpu_sequential_original":                            
                        
                            in_file_result = exec_command(["-c","-i",os.path.abspath(os.path.join(dir_name,dataset)),"-o",os.path.join(dir_name,"temp.lz"),"-p","cpu_sequential", "-w","1"])
                            in_memory_result = exec_command(["-c","-i",os.path.abspath(os.path.join(dir_name,dataset)),"-o",os.path.join(dir_name,"temp.lz"),"-p","cpu_sequential", "-w","1","-m"])
                        elif plan == "cpu_sequential":                            
                            in_file_result = exec_command(["-c","-i",os.path.abspath(os.path.join(dir_name,dataset)),"-o",os.path.join(dir_name,"temp.lz"),"-p",plan,"-s", "-g" ,gpus_arg, "-w",str(THREADS_CPU[threads])])
                            in_memory_result = exec_command(["-c","-i",os.path.abspath(os.path.join(dir_name,dataset)),"-o",os.path.join(dir_name,"temp.lz"),"-p",plan,"-m","-s", "-g", gpus_arg,"-w",str(THREADS_CPU[threads])])
                        else:
                            in_file_result = exec_command(["-c","-i",os.path.abspath(os.path.join(dir_name,dataset)),"-o",os.path.join(dir_name,"temp.lz"),"-p",plan,"-s", "-g" ,gpus_arg])
                            in_memory_result = exec_command(["-c","-i",os.path.abspath(os.path.join(dir_name,dataset)),"-o",os.path.join(dir_name,"temp.lz"),"-p",plan,"-m","-s", "-g", gpus_arg ])

                        for result in [process_output(in_file_result),process_output(in_memory_result)]:
                            result.seq = i
                            result.plan = plan

                            result.threads = threads
                            result.dataset = dataset
                            insert_result(db,result)
                        
                        break
                    except Exception as err:
                        log("An error ocurred: {0}".format(err))
                        sleep(1)
                        log("Trying again for {0} time".format(retrial + 1))
            
            current = current +1
            log("PROGRESSO: {0:.2f}".format(current/total * 100))


    db.close()

def insert_result(db,item):
    cursor = db.cursor()
    cursor.execute("""INSERT INTO result( dataset ,
            mode ,
            plan ,
            seq ,
            input_file_size ,
            output_file_size  ,
            compress_ratio ,
            input_processing,
            output_processing,
            total_time,
            gpu_host_to_device ,
            gpu_device_to_host ,
            gpu_kernel ,
            finding_match,
            threads,
            current_date,
            current_time ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,date('now'),time('now')) """,(
                item.dataset,
                item.file_statistics,
                item.plan,
                item.seq,
                item.input_file_size,
                item.output_file_size,
                item.compress_ratio,
                item.input_processing,
                item.output_processing,
                item.total_time,
                item.gpu_host_to_device,
                item.gpu_device_to_host,
                item.gpu_kernel,
                item.finding_match,
                item.threads,
            ))
    db.commit()
        

def exec_command(command):
    app_path = os.path.join(EXE_PATH,APP_FILE)
    log("CMD: "+" ".join(command))
    # return ""
    process = Popen([app_path] + command, stdout=PIPE, cwd=EXE_PATH)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        raise Exception("Wrong result for command '%s', output is %s " % (" ".join(command),output))
    return output.decode('ascii')
def process_output(text):
    result = ItemResult()
    result.plan = find_item(r"Selected Plan: ([A-Za-z_]+)",text,"")
    result.file_statistics = find_item( r"File statistics: ([A-Za-z]+)",text,"")
    result.gpu_host_to_device = float(find_item(r"timeSpentOnMemoryHostToDevice: ([\d\.]+) seconds",text,0))
    result.gpu_device_to_host = float(find_item(r"timeSpentOnMemoryDeviceToHost: ([\d\.]+) seconds",text,0))
    result.gpu_kernel = float(find_item(r"timeSpentOnKernel: ([\d\.]+) seconds",text,0))
    result.total_time = float(find_item(r"Total time: ([\d\.]+) seconds",text,0))
    result.finding_match = float(find_item(r"Finding match: ([\d\.]+) seconds",text,0))
    result.input_file_size = float(find_item(r"Input file size: ([\d\.]+) MB",text,0))
    result.output_file_size = float(find_item(r"Output file size: ([\d\.]+) MB",text,0))
    result.compress_ratio = float(find_item(r"Compress ratio: ([\d\.]+)%",text,0))
    result.input_processing = float(find_item(r"Input processing: ([\d\.]+) MB",text,0))
    result.output_processing = float(find_item(r"Output processing: ([\d\.]+) MB",text,0))

    return result

    
def find_item(pattern, text, default):
    found = re.findall(pattern,text)
    if len(found) == 0:
        return default
    return found[0]
if __name__ == "__main__":
    main(argv)
    # print(vars(result))
