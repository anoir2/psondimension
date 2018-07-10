
Particle Swarm Optimization
============================
### Test results
______
NVIDIA GTX 755M
---

#### [INPUT 2 DIM, ONE MILION PARTICLES]
##### main_v2.cu
```
==4325== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.11%  35.351ms         9  3.9278ms  3.7610ms  4.3540ms  new_pos(ParticleSystem*)
                   34.67%  26.580ms         9  2.9533ms  2.7988ms  3.2970ms  new_vel(ParticleSystem*)
                   13.75%  10.542ms      1710  6.1650us  2.4320us  9.9840us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                    5.45%  4.1774ms         1  4.1774ms  4.1774ms  4.1774ms  init_particle(ParticleSystem*)
                    0.02%  16.000us        10  1.6000us  1.6000us  1.6000us  [CUDA memcpy DtoH]
                    0.00%  2.5920us         2  1.2960us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   49.50%  91.244ms         1  91.244ms  91.244ms  91.244ms  cudaMemcpyToSymbol
                   33.79%  62.290ms        27  2.3071ms  6.8340us  4.5001ms  cudaEventSynchronize
                    9.30%  17.150ms        42  408.34us  4.1730us  4.9848ms  cudaFree
                    4.02%  7.4032ms      1729  4.2810us  3.6110us  73.759us  cudaLaunchKernel
                    2.44%  4.5024ms        42  107.20us  6.3720us  371.95us  cudaMalloc
                    0.39%  717.69us        96  7.4750us     191ns  333.54us  cuDeviceGetAttribute
                    0.21%  384.00us         1  384.00us  384.00us  384.00us  cuDeviceGetName
                    0.20%  362.64us        11  32.967us  9.3540us  45.344us  cudaMemcpy
                    0.06%  103.72us        54  1.9200us  1.1730us  5.0970us  cudaEventRecord
                    0.05%  88.034us         1  88.034us  88.034us  88.034us  cuDeviceTotalMem
                    0.03%  51.012us        54     944ns     465ns  3.4880us  cudaEventCreate
                    0.02%  33.152us        27  1.2270us     883ns  3.2320us  cudaEventElapsedTime
                    0.00%  2.3630us         1  2.3630us  2.3630us  2.3630us  cuDeviceGetPCIBusId
                    0.00%  1.6870us         2     843ns     268ns  1.4190us  cuDeviceGetCount
                    0.00%     871ns         2     435ns     239ns     632ns  cuDeviceGet
```

##### main_v3.cu
```
==4345== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   36.33%  8.3660ms         3  2.7887ms  2.7239ms  2.9089ms  new_pos(ParticleSystem*)
                   25.97%  5.9795ms         3  1.9932ms  1.9859ms  1.9989ms  new_vel(ParticleSystem*)
                   19.48%  4.4850ms       660  6.7950us  3.6160us  8.4480us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                   18.19%  4.1889ms         1  4.1889ms  4.1889ms  4.1889ms  init_particle(ParticleSystem*)
                    0.03%  6.3690us         4  1.5920us  1.5680us  1.6010us  [CUDA memcpy DtoH]
                    0.01%  2.5920us         2  1.2960us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   75.35%  90.541ms         1  90.541ms  90.541ms  90.541ms  cudaMemcpyToSymbol
                   11.99%  14.413ms         9  1.6014ms  7.0080us  2.9174ms  cudaEventSynchronize
                    7.55%  9.0684ms        14  647.74us  4.2450us  5.0742ms  cudaFree
                    2.73%  3.2823ms       667  4.9210us  3.9430us  26.426us  cudaLaunchKernel
                    1.18%  1.4127ms        14  100.91us  6.9120us  358.73us  cudaMalloc
                    0.80%  959.84us        96  9.9980us     236ns  429.29us  cuDeviceGetAttribute
                    0.13%  157.89us         5  31.578us  14.773us  41.634us  cudaMemcpy
                    0.11%  129.71us         1  129.71us  129.71us  129.71us  cuDeviceTotalMem
                    0.10%  123.95us         1  123.95us  123.95us  123.95us  cuDeviceGetName
                    0.03%  35.354us        18  1.9640us  1.2500us  3.1920us  cudaEventRecord
                    0.01%  17.282us        18     960ns     500ns  2.4050us  cudaEventCreate
                    0.01%  11.042us         9  1.2260us     931ns  1.8000us  cudaEventElapsedTime
                    0.00%  2.5460us         1  2.5460us  2.5460us  2.5460us  cuDeviceGetPCIBusId
                    0.00%  2.1410us         2  1.0700us     559ns  1.5820us  cuDeviceGetCount
                    0.00%  1.1360us         2     568ns     283ns     853ns  cuDeviceGet
```

##### main_v4.cu
```
==4363== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.24%  11.155ms         3  3.7182ms  3.6102ms  3.9020ms  new_vel_pos(ParticleSystem*)
                   24.03%  4.1727ms         1  4.1727ms  4.1727ms  4.1727ms  init_particle(ParticleSystem*)
                   11.67%  2.0264ms        12  168.87us  3.5200us  499.77us  find_min_fitness_parallel(ParticleSystem*, fitness_pos const *, fitness_pos*, int, int, int)
                    0.05%  8.8320us         5  1.7660us  1.7280us  1.8560us  [CUDA memcpy DtoH]
                    0.01%  2.5920us         2  1.2960us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   88.70%  173.92ms         1  173.92ms  173.92ms  173.92ms  cudaMemcpyToSymbol
                    5.71%  11.193ms         6  1.8656ms  6.6680us  3.9119ms  cudaEventSynchronize
                    3.70%  7.2615ms        14  518.68us  4.9450us  4.6107ms  cudaFree
                    0.68%  1.3365ms        14  95.462us  6.3390us  311.89us  cudaMalloc
                    0.45%  880.31us         6  146.72us  9.6470us  708.04us  cudaMemcpy
                    0.39%  757.84us        96  7.8940us     279ns  353.21us  cuDeviceGetAttribute
                    0.20%  396.02us         1  396.02us  396.02us  396.02us  cuDeviceGetName
                    0.09%  172.02us        16  10.751us  6.1430us  20.917us  cudaLaunchKernel
                    0.05%  92.966us         1  92.966us  92.966us  92.966us  cuDeviceTotalMem
                    0.02%  34.528us        12  2.8770us  1.2960us  11.518us  cudaEventRecord
                    0.01%  17.929us         6  2.9880us  1.1040us  10.751us  cudaEventElapsedTime
                    0.01%  14.350us        12  1.1950us     647ns  3.0320us  cudaEventCreate
                    0.00%  2.4750us         1  2.4750us  2.4750us  2.4750us  cuDeviceGetPCIBusId
                    0.00%  1.9150us         2     957ns     433ns  1.4820us  cuDeviceGetCount
                    0.00%     970ns         2     485ns     314ns     656ns  cuDeviceGet
```

##### main_v5.cu
```
==4379== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   35.04%  1.1314ms         4  282.86us  8.6720us  557.31us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                   26.92%  869.31us         1  869.31us  869.31us  869.31us  calc_fitness(void)
                   23.13%  747.07us         1  747.07us  747.07us  747.07us  init_particle(void)
                   14.16%  457.44us         1  457.44us  457.44us  457.44us  new_vel_pos(void)
                    0.38%  12.320us        10  1.2320us  1.2160us  1.3440us  [CUDA memcpy HtoD]
                    0.37%  11.872us         6  1.9780us  1.4400us  2.3040us  [CUDA memcpy DtoH]
      API calls:   96.48%  128.29ms        12  10.691ms  4.5040us  126.87ms  cudaMalloc
                    1.47%  1.9539ms         4  488.48us  6.3360us  1.2480ms  cudaFree
                    1.02%  1.3535ms         3  451.17us  6.7600us  879.34us  cudaEventSynchronize
                    0.55%  731.62us        96  7.6210us     208ns  417.60us  cuDeviceGetAttribute
                    0.17%  222.60us         6  37.099us  12.177us  85.566us  cudaMemcpy
                    0.09%  121.18us         1  121.18us  121.18us  121.18us  cuDeviceTotalMem
                    0.07%  96.217us        10  9.6210us  8.9740us  13.700us  cudaMemcpyToSymbol
                    0.07%  94.092us         1  94.092us  94.092us  94.092us  cuDeviceGetName
                    0.06%  75.705us         7  10.815us  5.6710us  20.746us  cudaLaunchKernel
                    0.01%  12.957us         6  2.1590us  1.4950us  3.0810us  cudaEventRecord
                    0.01%  7.5630us         6  1.2600us     606ns  3.0520us  cudaEventCreate
                    0.00%  4.2630us         3  1.4210us  1.1720us  1.6020us  cudaEventElapsedTime
                    0.00%  2.7110us         2  1.3550us     746ns  1.9650us  cuDeviceGetCount
                    0.00%  2.0410us         1  2.0410us  2.0410us  2.0410us  cuDeviceGetPCIBusId
                    0.00%  1.7980us         8     224ns     128ns     400ns  cudaGetSymbolAddress
                    0.00%  1.2580us         2     629ns     451ns     807ns  cuDeviceGet
```

##### main_v6.cu
```
==4398== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   37.39%  5.6632ms        20  283.16us  7.4560us  559.04us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                   30.06%  4.5519ms         9  505.77us  495.36us  513.67us  calc_fitness(void)
                   27.12%  4.1067ms         9  456.30us  454.30us  458.21us  new_vel_pos(void)
                    4.97%  752.19us         1  752.19us  752.19us  752.19us  init_particle(void)
                    0.38%  58.144us        30  1.9380us     864ns  2.2720us  [CUDA memcpy DtoH]
                    0.08%  12.320us        10  1.2320us  1.2160us  1.3440us  [CUDA memcpy HtoD]
      API calls:   85.56%  116.41ms        28  4.1576ms  4.6180us  113.42ms  cudaMalloc
                    6.82%  9.2791ms        27  343.67us  4.1870us  872.80us  cudaEventSynchronize
                    5.56%  7.5682ms        20  378.41us  5.6780us  1.2554ms  cudaFree
                    0.82%  1.1097ms        30  36.989us  11.663us  86.673us  cudaMemcpy
                    0.58%  784.68us        96  8.1730us     271ns  344.02us  cuDeviceGetAttribute
                    0.30%  408.06us        39  10.463us  4.9700us  21.695us  cudaLaunchKernel
                    0.09%  117.87us         1  117.87us  117.87us  117.87us  cuDeviceTotalMem
                    0.07%  97.551us        54  1.8060us  1.1250us  3.2120us  cudaEventRecord
                    0.07%  96.716us         1  96.716us  96.716us  96.716us  cuDeviceGetName
                    0.07%  95.766us        10  9.5760us  8.8680us  13.905us  cudaMemcpyToSymbol
                    0.04%  49.552us        54     917ns     516ns  2.8830us  cudaEventCreate
                    0.02%  30.792us        27  1.1400us     812ns  2.1070us  cudaEventElapsedTime
                    0.01%  8.0840us        40     202ns     131ns     422ns  cudaGetSymbolAddress
                    0.00%  2.5620us         1  2.5620us  2.5620us  2.5620us  cuDeviceGetPCIBusId
                    0.00%  2.3180us         2  1.1590us     489ns  1.8290us  cuDeviceGetCount
                    0.00%  1.0630us         2     531ns     339ns     724ns  cuDeviceGet
```

