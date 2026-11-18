import re, os, subprocess, tempfile, json
import docker, time
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

# resource tracking libraries
import time # to measure runtime
import psutil # to measure memory
from codecarbon import EmissionsTracker, OfflineEmissionsTracker # to measure energy/CO2eq (for hardware you control)
from ecologits import EcoLogits # to estimate energy and carbon for API calls to LLMs


# importing variables from the config file
from supercode.src.configs.parse_config import *


# ----------------------------------------------------------------------------------------------
def resource_tracking(func, **args):
    """
    returns the function results along with resources tracked by CodeCarbon and tracemalloc.

    CodeCarbon reads/estimates the direct electricity consumption of compute components (CPU, GPU and RAM)
    and multiplies the resulting energy by the carbon intensity of the local electricity grid.
    Excluded: disk I/O, network transfers, displays, cooling and other peripherals,
    power supply and datacenter overhead and life-cycle (embodied) emissions of the hardware.
    """

    def process_memory():
        # https://psutil.io/shell-equivalents/
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        # cpu_perc = process.cpu_percent(interval=1.0)
        return mem_info.rss

    # track energy and carbon emissions, tracker options:
    # - country_iso_code="NLD" --> must be the 3 letters iso code
    # - rapl_include_dram=True (default is false)
    # - measure_power_secs=15 (default is 15)
    tracker = EmissionsTracker() # OfflineEmissionsTracker(country_iso_code="NLD")
    tracker.start()

    # track memory allocation
    mem_start = process_memory()

    # track runtime
    tot_start = time.perf_counter() # total runtime from performance counters (more accurate)
    cpu_start = time.process_time() # cpu runtime

    func_results = func(**args)

    tot_end = time.perf_counter() # total runtime from performance counters (more accurate)
    cpu_end = time.process_time() # cpu runtime
    psutil_memory = (process_memory()-mem_start)/1048576 # conversion from bytes to MiB
    emissions = tracker.stop() # kg CO2eq (--> energy * PUE)
    energy = tracker._total_energy # kWh
    tot_time = tot_end - tot_start
    cpu_time = cpu_end - cpu_start

    if verbose>1: print(f">> resource tracking for {func.__name__}:"
        f"\n  - total runtime = {tot_time} seconds"
        f"\n  - CPU runtime = {cpu_time} seconds"
        f"\n  - energy = {energy} kWh"
        f"\n  - emissions = {emissions} kg CO2eq (= energy*PUE)"
        f"\n  - psutil_memory = {psutil_memory} MiB"
        )
    resources = {
        "tot_time": float(tot_time),
        "cpu_time": float(cpu_time),
        "energy": float(energy),
        "emissions": float(emissions),
        "psutil_memory": float(psutil_memory)
    }
    return func_results, resources

# ----------------------------------------------------------------------------------------------
def docker_container(command: Literal['activate', 'deactivate', 'delete']):
    """Activate or deactivate the sandbox docker container for running code."""
    image = docker_args["image"]
    container_name = docker_args["container_name"]
    workspace_path = os.path.join(os.getcwd(), docker_args["workspace"])
    #print("workspace =", workspace_path)

    if verbose>1: print(f">> about to {command} docker container '{container_name}'")
    # for debugging
    # if verbose>2:
    #     print("----------------------------------------------------------------------------------------------\n", subprocess.run(
    #         # 'docker ps' returns info of all (-a) containers filtered (-f) by name
    #         ["docker", "ps", "-a", "-f", f"name={container_name}"],
    #         capture_output=True, text=True, check=True
    #     ).stdout.strip(), "\n", sep="")

    # --------------------------------------------------------------------------------------
    def execute_docker_command(command: Literal['build', 'create', 'activate', 'deactivate', 'delete']):
        if command=="build":
            # build the image for the container (docker build) based on the Dockerfile
            if verbose>1: print(">> building image")
            subprocess.run([
                "docker", "build", "-t", image, docker_args["docker_dir"]
            ], check=True)

        elif command=="create":
            # create a new container (docker run)
            if verbose>1: print(">> creating container")
            subprocess.run([
                "docker", "run", "-d", "-it", # -d for detached mode (run in background and print container ID), -it for Interactive TTY session
                "-m", docker_args["mem_limit"],  # memory limit, defined in config
                "--name", container_name,  # fixed name for the container, defined in config
                "--mount", f"type=bind,src={workspace_path},dst=/workspace",  # mount disk to save/access files
                image  # select image for the docker container
            ], check=True)
            # other options:
            # --rm for auto-cleanup on stop (very slow if done often)
            # -v for binding the workspace directory as volume

        elif command == "activate":
            # activate the container (docker start)
            if verbose>1: print(">> starting container")
            subprocess.run(["docker", "start", container_name], check=True)

        elif command=="deactivate":
            # deactivate the container if it is running (docker stop)
            if verbose>1: print(">> stopping container")
            subprocess.run(["docker", "stop", container_name], check=True)

        elif command=="delete":
            # delete the container (docker rm)
            if verbose>1: print(">> removing container")
            subprocess.run(["docker", "rm", container_name], check=True)

        else:
            print(f"WARNING: unknown command {command} will be ignored")
    # --------------------------------------------------------------------------------------
    try:
        # check if the container exists (active or not)
        exist_docker_id = subprocess.run(
            # 'docker ps' returns the IDs (-q) of all (-a) containers filtered (-f) by name
            ["docker", "ps", "-a", "-q", "-f", f"name={container_name}"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        # check if the container is active (this is a subset of the previous)
        active_docker_id = subprocess.run(
            # 'docker ps' returns the IDs (-q) of active containers filtered (-f) by name
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        image_in_mem = image in subprocess.run(["docker", "images"], capture_output=True, text=True, check=True).stdout.strip()

        # decision tree based on the command and whether the container exists/is running
        if exist_docker_id: # check if the container exists
            if active_docker_id: # check if is it active
                if command == 'deactivate' or command == 'delete':
                    execute_docker_command('deactivate')
                    if command == 'delete': execute_docker_command(command)
                # pass if command is activate
            else: # if the container exists, but it is not active
                if command == 'activate' or 'delete': execute_docker_command(command)
                # pass if command is deactivate
        else: # if the container doesn't exist
            if command == 'activate':
                if not image_in_mem: execute_docker_command('build') # build the image from the Dockerfile
                execute_docker_command('create') # then create the container from the image
                execute_docker_command('activate') # the created container automatically stops if no commands are given
            # pass if deactivate/delete

    except subprocess.CalledProcessError as e:
        print(f"ERROR: docker_container tried to {command} {container_name} but failed:\n{e}")

    # for debugging
    if verbose>2:
        print("\n", subprocess.run(
            # 'docker ps' returns info of all (-a) containers filtered (-f) by name
            ["docker", "ps", "-a", "-f", f"name={container_name}"],
            capture_output=True, text=True, check=True
        ).stdout.strip(), "\n----------------------------------------------------------------------------------------------\n", sep="")

# ------------------------------------------------------------------------------------
def get_docker_raw_stats():
    client = docker.from_env()
    containers = client.containers.list() #; print(containers)
    container = None ; stats = None
    try:
        for container in containers:
            # Get a single stats snapshot (stream=False returns one reading)
            stats = container.stats(stream=False)
            # print("\nstats vals:\n", json.dumps(stats, indent=4))

            if container.name == docker_args["container_name"] : break
            else: continue

        # convert time string to time in seconds (format: "2026-11-17T12:23:10.950959375Z")
        pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).(\d*)Z'
        match = re.search(pattern, stats["read"], re.DOTALL)

        t1 = time.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S") # convert formatted datatime string into time tuple
        t2 = time.mktime(t1) # convert time tuple into seconds since epoch
        epoch_time = t2 + float("0."+match.group(2)) # add fraction of seconds (direct conversion is not supported)

        # Extract the fields we care about
        return {
            "name": container.name,
            "epoch_time": epoch_time,
            "cpu_total": stats["cpu_stats"]["cpu_usage"]["total_usage"]/1048576, # in MB
            # NOTE: "cpu_total" is the sum of "usage_in_kernelmode" and "usage_in_usermode"
            "memory_usage": stats["memory_stats"].get("usage", 0)/1048576, # in MiB
            # "memory_limit": stats["memory_stats"].get("limit", 0),
            # "network_rx": sum(v["rx_bytes"] for v in stats.get("networks", {}).values()),
            # "network_tx": sum(v["tx_bytes"] for v in stats.get("networks", {}).values()),
        }
    except Exception as e:
        print(f"\nAn error occurred in get_docker_stats:\n{e}\n")
        return None

# ----------------------------------------------------------------------------------------------
def get_docker_delta_stats(stats_0, stats_f):
    for key in stats_f.keys():
        if key == "name": continue
        stats_f[key] -= stats_0[key]
    return stats_f

# ----------------------------------------------------------------------------------------------
def extract_runtime(code_output):
    pattern = r'real\s(\d+)m(\d+\.\d+)s\suser\s(\d+)m(\d+\.\d+)s\ssys\s(\d+)m(\d+\.\d+)s\s'
    match = re.search(pattern, code_output) # there should be only one match
    # time <command> output:
    # <command output>
    # real	0m0.004s
    # user	0m0.001s
    # sys	0m0.004s

    real_t = float(match.group(1))*60+float(match.group(2))
    user_t = float(match.group(3))*60+float(match.group(4))
    sys_t  = float(match.group(5))*60+float(match.group(6))

    runtime = {
        "real": real_t, # clock time (including I/O and time slices used by other processes)
        "user": user_t, # process time for the current job only
        "sys":  sys_t,  # kernel time for the current job only (this doesn't include library code --> user)
        "cpu":  user_t+sys_t, # cpu time, is <=real with single-thread and can be >real with multi-thread
        "util": (user_t+sys_t)/real_t # cpu utilization, note this can be >1 with multi-threading
    }
    if verbose>1: print(">> code runtime:", runtime)
    return runtime, match.group(0)
