
Particle Swarm Optimization
============================
### Test results - NVIDIA GTX 860M
______

### [INPUT 2 DIM, ONE MILION PARTICLES]
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
______
### [INPUT 8 DIM, ONE MILION PARTICLES]
##### main_v2.cu
```
==4497== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.21%  28.193ms         7  4.0275ms  3.7606ms  4.3411ms  new_pos(ParticleSystem*)
                   33.18%  20.247ms         7  2.8925ms  2.8027ms  3.0226ms  new_vel(ParticleSystem*)
                   13.76%  8.3933ms      1368  6.1350us  2.4320us  9.3440us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                    6.83%  4.1648ms         1  4.1648ms  4.1648ms  4.1648ms  init_particle(ParticleSystem*)
                    0.02%  12.832us         8  1.6040us  1.6000us  1.6320us  [CUDA memcpy DtoH]
                    0.00%  2.5920us         2  1.2960us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   55.60%  94.369ms         1  94.369ms  94.369ms  94.369ms  cudaMemcpyToSymbol
                   28.73%  48.767ms        21  2.3222ms  6.7730us  4.5055ms  cudaEventSynchronize
                    9.17%  15.574ms        34  458.05us  4.3340us  4.9380ms  cudaFree
                    3.40%  5.7631ms      1383  4.1670us  3.6390us  21.320us  cudaLaunchKernel
                    2.02%  3.4222ms        34  100.65us  6.4980us  256.65us  cudaMalloc
                    0.58%  979.60us        96  10.204us     250ns  436.34us  cuDeviceGetAttribute
                    0.18%  304.95us         9  33.883us  9.9780us  45.916us  cudaMemcpy
                    0.17%  291.87us         1  291.87us  291.87us  291.87us  cuDeviceGetName
                    0.07%  122.09us         1  122.09us  122.09us  122.09us  cuDeviceTotalMem
                    0.05%  81.801us        42  1.9470us  1.1420us  13.637us  cudaEventRecord
                    0.02%  37.789us        42     899ns     452ns  3.6070us  cudaEventCreate
                    0.01%  22.581us        21  1.0750us     878ns  1.6230us  cudaEventElapsedTime
                    0.00%  2.5960us         1  2.5960us  2.5960us  2.5960us  cuDeviceGetPCIBusId
                    0.00%  2.2230us         2  1.1110us     623ns  1.6000us  cuDeviceGetCount
                    0.00%     985ns         2     492ns     371ns     614ns  cuDeviceGet
```

##### main_v3.cu
```
==4509== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   39.24%  13.708ms         5  2.7416ms  2.6462ms  2.9098ms  new_pos(ParticleSystem*)
                   29.55%  10.323ms         5  2.0646ms  1.9800ms  2.3988ms  new_vel(ParticleSystem*)
                   19.21%  6.7113ms       990  6.7790us  3.5840us  8.1280us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                   11.96%  4.1771ms         1  4.1771ms  4.1771ms  4.1771ms  init_particle(ParticleSystem*)
                    0.03%  9.8880us         6  1.6480us  1.6000us  1.8880us  [CUDA memcpy DtoH]
                    0.01%  2.5600us         2  1.2800us  1.2160us  1.3440us  [CUDA memcpy HtoD]
      API calls:   66.20%  87.544ms         1  87.544ms  87.544ms  87.544ms  cudaMemcpyToSymbol
                   18.27%  24.158ms        15  1.6105ms  6.9060us  2.9200ms  cudaEventSynchronize
                    9.65%  12.767ms        20  638.34us  4.4330us  5.1316ms  cudaFree
                    3.37%  4.4592ms      1001  4.4540us  3.7310us  24.317us  cudaLaunchKernel
                    1.29%  1.7065ms        20  85.326us  6.9570us  248.81us  cudaMalloc
                    0.59%  783.91us        96  8.1650us     214ns  365.19us  cuDeviceGetAttribute
                    0.30%  399.14us         1  399.14us  399.14us  399.14us  cuDeviceGetName
                    0.15%  200.56us         7  28.651us  9.3880us  40.507us  cudaMemcpy
                    0.08%  99.908us         1  99.908us  99.908us  99.908us  cuDeviceTotalMem
                    0.05%  60.462us        30  2.0150us  1.1220us  3.4560us  cudaEventRecord
                    0.03%  39.551us        30  1.3180us     451ns  11.155us  cudaEventCreate
                    0.02%  20.089us        15  1.3390us     978ns  2.0550us  cudaEventElapsedTime
                    0.00%  2.4590us         1  2.4590us  2.4590us  2.4590us  cuDeviceGetPCIBusId
                    0.00%  1.7670us         2     883ns     392ns  1.3750us  cuDeviceGetCount
                    0.00%     903ns         2     451ns     229ns     674ns  cuDeviceGet
```

##### main_v4.cu
```
==4522== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.43%  14.536ms         4  3.6341ms  3.4666ms  3.8526ms  new_vel_pos(ParticleSystem*)
                   19.60%  4.1626ms         1  4.1626ms  4.1626ms  4.1626ms  init_particle(ParticleSystem*)
                   11.91%  2.5299ms        15  168.66us  3.5200us  500.90us  find_min_fitness_parallel(ParticleSystem*, fitness_pos const *, fitness_pos*, int, int, int)
                    0.05%  10.560us         6  1.7600us  1.7280us  1.8560us  [CUDA memcpy DtoH]
                    0.01%  2.5920us         2  1.2960us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   77.74%  91.548ms         1  91.548ms  91.548ms  91.548ms  cudaMemcpyToSymbol
                   12.40%  14.605ms         8  1.8257ms  6.6930us  3.8636ms  cudaEventSynchronize
                    6.72%  7.9151ms        17  465.60us  4.8240us  4.5872ms  cudaFree
                    1.37%  1.6098ms        17  94.693us  6.1280us  309.87us  cudaMalloc
                    0.78%  920.67us         7  131.52us  9.4870us  694.38us  cudaMemcpy
                    0.57%  666.59us        96  6.9430us     238ns  300.22us  cuDeviceGetAttribute
                    0.18%  206.31us        20  10.315us  5.4740us  21.099us  cudaLaunchKernel
                    0.09%  109.49us         1  109.49us  109.49us  109.49us  cuDeviceTotalMem
                    0.09%  103.14us         1  103.14us  103.14us  103.14us  cuDeviceGetName
                    0.03%  30.567us        16  1.9100us  1.2270us  3.8820us  cudaEventRecord
                    0.02%  26.503us        16  1.6560us     446ns  12.929us  cudaEventCreate
                    0.01%  10.575us         8  1.3210us     995ns  2.2640us  cudaEventElapsedTime
                    0.00%  2.3930us         1  2.3930us  2.3930us  2.3930us  cuDeviceGetPCIBusId
                    0.00%  1.9010us         2     950ns     427ns  1.4740us  cuDeviceGetCount
                    0.00%  1.1060us         2     553ns     308ns     798ns  cuDeviceGet
```

##### main_v5.cu
```
==4535== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   34.95%  1.1277ms         4  281.92us  8.6720us  555.27us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                   27.11%  874.70us         1  874.70us  874.70us  874.70us  calc_fitness(void)
                   23.09%  745.07us         1  745.07us  745.07us  745.07us  init_particle(void)
                   14.09%  454.57us         1  454.57us  454.57us  454.57us  new_vel_pos(void)
                    0.38%  12.352us        10  1.2350us  1.2160us  1.3760us  [CUDA memcpy HtoD]
                    0.37%  11.808us         6  1.9680us  1.4400us  2.2720us  [CUDA memcpy DtoH]
      API calls:   95.35%  91.087ms        12  7.5906ms  4.9600us  89.675ms  cudaMalloc
                    2.03%  1.9400ms         4  485.01us  6.0640us  1.2433ms  cudaFree
                    1.42%  1.3529ms         3  450.98us  6.8030us  882.25us  cudaEventSynchronize
                    0.56%  538.29us        96  5.6070us     239ns  241.68us  cuDeviceGetAttribute
                    0.23%  221.53us         6  36.921us  11.623us  85.569us  cudaMemcpy
                    0.10%  93.197us        10  9.3190us  8.8600us  12.751us  cudaMemcpyToSymbol
                    0.10%  93.196us         1  93.196us  93.196us  93.196us  cuDeviceTotalMem
                    0.10%  92.131us         1  92.131us  92.131us  92.131us  cuDeviceGetName
                    0.09%  85.838us         7  12.262us  6.0380us  21.306us  cudaLaunchKernel
                    0.01%  12.525us         6  2.0870us  1.1610us  3.3370us  cudaEventRecord
                    0.01%  6.0780us         6  1.0130us     493ns  2.2470us  cudaEventCreate
                    0.00%  3.6190us         3  1.2060us     958ns  1.6000us  cudaEventElapsedTime
                    0.00%  2.4670us         1  2.4670us  2.4670us  2.4670us  cuDeviceGetPCIBusId
                    0.00%  2.0970us         2  1.0480us     339ns  1.7580us  cuDeviceGetCount
                    0.00%  1.5260us         8     190ns     132ns     272ns  cudaGetSymbolAddress
                    0.00%     909ns         2     454ns     287ns     622ns  cuDeviceGet
```

##### main_v6.cu
```
==4548== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   37.51%  4.5125ms        16  282.03us  7.4240us  556.19us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                   29.36%  3.5324ms         7  504.62us  498.85us  508.74us  calc_fitness(void)
                   26.48%  3.1858ms         7  455.11us  453.02us  457.79us  new_vel_pos(void)
                    6.15%  739.68us         1  739.68us  739.68us  739.68us  init_particle(void)
                    0.39%  47.136us        24  1.9640us  1.3440us  2.4000us  [CUDA memcpy DtoH]
                    0.10%  12.320us        10  1.2320us  1.2160us  1.3440us  [CUDA memcpy HtoD]
      API calls:   84.74%  91.153ms        24  3.7980ms  4.7520us  88.534ms  cudaMalloc
                    6.76%  7.2686ms        21  346.13us  6.6440us  852.27us  cudaEventSynchronize
                    5.90%  6.3454ms        16  396.59us  5.5000us  1.2334ms  cudaFree
                    0.91%  983.90us        24  40.995us  11.920us  176.13us  cudaMemcpy
                    0.67%  721.28us        96  7.5130us     194ns  337.82us  cuDeviceGetAttribute
                    0.36%  382.65us         1  382.65us  382.65us  382.65us  cuDeviceGetName
                    0.33%  351.81us        31  11.348us  5.2400us  31.453us  cudaLaunchKernel
                    0.09%  94.842us        10  9.4840us  8.7690us  13.229us  cudaMemcpyToSymbol
                    0.09%  94.619us        42  2.2520us  1.0800us  11.886us  cudaEventRecord
                    0.08%  87.573us         1  87.573us  87.573us  87.573us  cuDeviceTotalMem
                    0.03%  36.092us        42     859ns     430ns  2.4880us  cudaEventCreate
                    0.02%  23.325us        21  1.1100us     826ns  1.8410us  cudaEventElapsedTime
                    0.02%  16.394us        32     512ns     123ns  10.221us  cudaGetSymbolAddress
                    0.00%  2.6860us         1  2.6860us  2.6860us  2.6860us  cuDeviceGetPCIBusId
                    0.00%  2.3370us         2  1.1680us     893ns  1.4440us  cuDeviceGetCount
                    0.00%  1.1500us         2     575ns     356ns     794ns  cuDeviceGet
```
______
### [INPUT 16 DIM, ONE MILION PARTICLES]
##### main_v2.cu
```
==5799== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.16%  46.0727s      1502  30.674ms  30.508ms  35.821ms  new_vel(ParticleSystem*)
                   34.62%  25.2532s      1501  16.824ms  16.560ms  21.953ms  new_pos(ParticleSystem*)
                    2.19%  1.59612s    256842  6.2140us  2.2400us  12.256us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                    0.03%  21.663ms         1  21.663ms  21.663ms  21.663ms  init_particle(ParticleSystem*)
                    0.00%  2.9142ms      1502  1.9400us  1.2480us  5.9840us  [CUDA memcpy DtoH]
                    0.00%  2.5600us         2  1.2800us  1.1840us  1.3760us  [CUDA memcpy HtoD]
      API calls:   95.17%  71.3643s      4504  15.845ms  2.9410us  35.834ms  cudaEventSynchronize
                    2.06%  1.54851s    259847  5.9590us  3.9900us  399.43us  cudaLaunchKernel
                    1.69%  1.26700s      6008  210.89us  4.2240us  22.421ms  cudaFree
                    0.86%  642.11ms      6010  106.84us  6.4770us  1.6781ms  cudaMalloc
                    0.12%  90.105ms         1  90.105ms  90.105ms  90.105ms  cudaMemcpyToSymbol
                    0.05%  35.412ms      1503  23.561us  9.6140us  45.976us  cudaMemcpy
                    0.03%  21.124ms      9010  2.3440us  1.1940us  8.8280us  cudaEventRecord
                    0.01%  10.901ms      9010  1.2090us     450ns  55.479us  cudaEventCreate
                    0.01%  8.2630ms      4504  1.8340us     879ns  379.95us  cudaEventElapsedTime
                    0.00%  883.20us        96  9.1990us     250ns  410.19us  cuDeviceGetAttribute
                    0.00%  402.78us         1  402.78us  402.78us  402.78us  cuDeviceGetName
                    0.00%  112.16us         1  112.16us  112.16us  112.16us  cuDeviceTotalMem
                    0.00%  2.6340us         1  2.6340us  2.6340us  2.6340us  cuDeviceGetPCIBusId
                    0.00%  2.4030us         2  1.2010us     421ns  1.9820us  cuDeviceGetCount
                    0.00%     991ns         2     495ns     312ns     679ns  cuDeviceGet
```

##### main_v3.cu
```
==4966== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.61%  5.86732s       139  42.211ms  39.833ms  44.945ms  new_vel(ParticleSystem*)
                   41.56%  4.30721s       139  30.987ms  26.907ms  37.443ms  new_pos(ParticleSystem*)
                    1.48%  153.89ms     23100  6.6610us  3.6800us  9.1200us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                    0.34%  35.632ms         1  35.632ms  35.632ms  35.632ms  init_particle(ParticleSystem*)
                    0.00%  312.58us       140  2.2320us  1.7280us  2.6880us  [CUDA memcpy DtoH]
                    0.00%  2.0800us         2  1.0400us     608ns  1.4720us  [CUDA memcpy HtoD]
      API calls:   91.65%  10.6568s       417  25.556ms  6.8680us  45.906ms  cudaEventSynchronize
                    3.59%  417.76ms       422  989.96us  7.1610us  37.420ms  cudaFree
                    2.27%  264.41ms     23379  11.309us  6.0330us  684.91us  cudaLaunchKernel
                    1.90%  221.40ms         1  221.40ms  221.40ms  221.40ms  cudaMemcpyToSymbol
                    0.47%  54.974ms       422  130.27us  11.558us  1.1126ms  cudaMalloc
                    0.04%  4.4028ms       141  31.225us  25.041us  36.978us  cudaMemcpy
                    0.03%  3.9472ms       834  4.7320us  2.2350us  13.518us  cudaEventRecord
                    0.02%  1.9897ms       834  2.3850us     967ns  5.7470us  cudaEventCreate
                    0.01%  1.3117ms       417  3.1450us  1.7250us  8.2790us  cudaEventElapsedTime
                    0.01%  759.68us        96  7.9130us     399ns  325.93us  cuDeviceGetAttribute
                    0.00%  106.99us         1  106.99us  106.99us  106.99us  cuDeviceTotalMem
                    0.00%  87.812us         1  87.812us  87.812us  87.812us  cuDeviceGetName
                    0.00%  5.5640us         1  5.5640us  5.5640us  5.5640us  cuDeviceGetPCIBusId
                    0.00%  2.5610us         2  1.2800us     844ns  1.7170us  cuDeviceGetCount
                    0.00%  1.3350us         2     667ns     525ns     810ns  cuDeviceGet
```

##### main_v4.cu
```
==4980== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.89%  12.5028s       168  74.422ms  72.471ms  78.924ms  new_vel_pos(ParticleSystem*)
                    0.80%  101.02ms       507  199.25us  3.6800us  1.7688ms  find_min_fitness_parallel(ParticleSystem*, fitness_pos const *, fitness_pos*, int, int, int)
                    0.31%  39.251ms         1  39.251ms  39.251ms  39.251ms  init_particle(ParticleSystem*)
                    0.00%  341.60us       170  2.0090us  1.7600us  2.4320us  [CUDA memcpy DtoH]
                    0.00%  2.9440us         2  1.4720us  1.4720us  1.4720us  [CUDA memcpy HtoD]
      API calls:   94.76%  12.6703s       336  37.709ms  3.5000us  80.017ms  cudaEventSynchronize
                    3.48%  465.36ms       509  914.27us  6.6180us  42.005ms  cudaFree
                    1.26%  168.72ms         1  168.72ms  168.72ms  168.72ms  cudaMemcpyToSymbol
                    0.36%  48.130ms       509  94.558us  8.6940us  815.33us  cudaMalloc
                    0.07%  9.6725ms       676  14.308us  8.2960us  28.417us  cudaLaunchKernel
                    0.04%  4.7142ms       171  27.568us  13.672us  712.24us  cudaMemcpy
                    0.01%  1.9130ms       672  2.8460us  1.6660us  8.5970us  cudaEventRecord
                    0.01%  1.0098ms       672  1.5020us     688ns  4.2160us  cudaEventCreate
                    0.00%  627.87us       336  1.8680us  1.3310us  3.9560us  cudaEventElapsedTime
                    0.00%  578.70us        96  6.0280us     162ns  259.67us  cuDeviceGetAttribute
                    0.00%  78.436us         1  78.436us  78.436us  78.436us  cuDeviceTotalMem
                    0.00%  68.473us         1  68.473us  68.473us  68.473us  cuDeviceGetName
                    0.00%  2.9370us         1  2.9370us  2.9370us  2.9370us  cuDeviceGetPCIBusId
                    0.00%  1.2140us         2     607ns     302ns     912ns  cuDeviceGetCount
                    0.00%     587ns         2     293ns     212ns     375ns  cuDeviceGet
```

##### main_v5.cu
```
==4996== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.41%  48.874ms        20  2.4437ms  2.1431ms  2.9917ms  calc_fitness(void)
                   37.83%  37.051ms         1  37.051ms  37.051ms  37.051ms  init_particle(void)
                    7.30%  7.7736ms        20  388.68us  362.38us  402.18us  new_vel_pos(void)
                    5.40%  2.2765ms         4  569.13us  25.792us  1.7060ms  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.03%  13.664us         6  2.2770us  1.5040us  4.4480us  [CUDA memcpy DtoH]
                    0.03%  12.640us        10  1.2640us  1.1840us  1.4720us  [CUDA memcpy HtoD]
      API calls:   81.31%  248.70ms        12  20.725ms  6.5260us  246.87ms  cudaMalloc
                   13.55%  41.462ms         4  10.366ms  8.5670us  39.825ms  cudaFree
                    3.19%  9.7512ms        10  975.12us  16.988us  1.1565ms  cudaMemcpyToSymbol
                    1.59%  4.8785ms         3  1.6262ms  6.2980us  3.6214ms  cudaEventSynchronize
                    0.20%  597.12us        96  6.2190us     211ns  261.73us  cuDeviceGetAttribute
                    0.07%  205.57us         6  34.261us  13.267us  74.588us  cudaMemcpy
                    0.03%  98.474us         7  14.067us  7.8670us  24.156us  cudaLaunchKernel
                    0.03%  80.258us         1  80.258us  80.258us  80.258us  cuDeviceTotalMem
                    0.02%  71.584us         1  71.584us  71.584us  71.584us  cuDeviceGetName
                    0.00%  15.272us         6  2.5450us  1.8290us  3.5570us  cudaEventRecord
                    0.00%  8.7250us         6  1.4540us     802ns  3.1180us  cudaEventCreate
                    0.00%  5.0690us         3  1.6890us  1.4010us  1.9990us  cudaEventElapsedTime
                    0.00%  2.5220us         1  2.5220us  2.5220us  2.5220us  cuDeviceGetPCIBusId
                    0.00%  2.2810us         8     285ns     193ns     493ns  cudaGetSymbolAddress
                    0.00%  1.5260us         2     763ns     430ns  1.0960us  cuDeviceGetCount
                    0.00%     765ns         2     382ns     292ns     473ns  cuDeviceGet
```

##### main_v6.cu
```
==5011== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.36%  710.94ms      2642  269.09us  15.456us  612.58us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                   34.26%  514.20ms      1320  389.55us  381.09us  425.76us  new_vel_pos(void)
                   15.35%  230.44ms      1320  174.57us  168.80us  218.98us  calc_fitness(void)
                    2.41%  36.233ms         1  36.233ms  36.233ms  36.233ms  init_particle(void)
                    0.61%  9.1920ms      3963  2.3190us  1.0240us  6.3360us  [CUDA memcpy DtoH]
                    0.00%  12.096us        10  1.2090us     608ns  1.5040us  [CUDA memcpy HtoD]
      API calls:   53.13%  3.48018s      3960  878.83us  3.2460us  2.1984ms  cudaEventSynchronize
                   32.14%  2.10514s      2642  796.80us  8.7590us  37.920ms  cudaFree
                   10.32%  676.24ms      2650  255.18us  7.4270us  266.79ms  cudaMalloc
                    2.20%  144.21ms      3963  36.388us  13.080us  807.22us  cudaMemcpy
                    1.29%  84.185ms      5283  15.934us  7.6690us  556.07us  cudaLaunchKernel
                    0.40%  26.338ms      7920  3.3250us  1.6960us  447.16us  cudaEventRecord
                    0.21%  14.039ms      7920  1.7720us     734ns  345.44us  cudaEventCreate
                    0.14%  9.1847ms        10  918.47us  20.212us  1.1919ms  cudaMemcpyToSymbol
                    0.13%  8.3715ms      3960  2.1140us  1.3650us  10.147us  cudaEventElapsedTime
                    0.03%  2.1240ms      5284     401ns     214ns  1.4270us  cudaGetSymbolAddress
                    0.01%  688.48us        96  7.1710us     213ns  304.42us  cuDeviceGetAttribute
                    0.00%  87.670us         1  87.670us  87.670us  87.670us  cuDeviceTotalMem
                    0.00%  76.798us         1  76.798us  76.798us  76.798us  cuDeviceGetName
                    0.00%  3.2420us         1  3.2420us  3.2420us  3.2420us  cuDeviceGetPCIBusId
                    0.00%  1.8430us         2     921ns     822ns  1.0210us  cuDeviceGetCount
                    0.00%     888ns         2     444ns     373ns     515ns  cuDeviceGet

```
______
### [INPUT 32 DIM, ONE MILION PARTICLES]
##### main_v2.cu
```
==5256== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.25%  55.5828s       527  105.47ms  58.843ms  142.19ms  new_vel(ParticleSystem*)
                   33.80%  28.7946s       527  54.639ms  30.771ms  74.385ms  new_pos(ParticleSystem*)
                    0.85%  719.84ms     90288  7.9720us  2.2400us  2.4223ms  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                    0.10%  87.802ms         1  87.802ms  87.802ms  87.802ms  init_particle(ParticleSystem*)
                    0.00%  1.1541ms       528  2.1850us  1.6960us  3.2000us  [CUDA memcpy DtoH]
                    0.00%  2.5280us         2  1.2640us     864ns  1.6640us  [CUDA memcpy HtoD]
      API calls:   96.32%  85.7375s      1581  54.230ms  4.1760us  143.30ms  cudaEventSynchronize
                    2.05%  1.82383s      2112  863.56us  7.2950us  90.177ms  cudaFree
                    0.92%  818.95ms     91344  8.9650us  5.6990us  905.70us  cudaLaunchKernel
                    0.37%  326.43ms      2114  154.41us  10.891us  1.2359ms  cudaMalloc
                    0.29%  254.22ms         1  254.22ms  254.22ms  254.22ms  cudaMemcpyToSymbol
                    0.02%  20.312ms       529  38.396us  20.748us  763.51us  cudaMemcpy
                    0.02%  15.073ms      3164  4.7630us  1.8210us  121.48us  cudaEventRecord
                    0.01%  7.1073ms      3164  2.2460us     754ns  39.800us  cudaEventCreate
                    0.01%  6.2747ms      1581  3.9680us  1.5230us  646.74us  cudaEventElapsedTime
                    0.00%  669.86us        96  6.9770us     195ns  297.35us  cuDeviceGetAttribute
                    0.00%  89.863us         1  89.863us  89.863us  89.863us  cuDeviceTotalMem
                    0.00%  73.987us         1  73.987us  73.987us  73.987us  cuDeviceGetName
                    0.00%  3.0300us         1  3.0300us  3.0300us  3.0300us  cuDeviceGetPCIBusId
                    0.00%  1.3290us         2     664ns     281ns  1.0480us  cuDeviceGetCount
                    0.00%     703ns         2     351ns     273ns     430ns  cuDeviceGet
```

##### main_v3.cu
```
==5282== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.57%  23.6257s       420  56.252ms  47.047ms  99.689ms  new_vel(ParticleSystem*)
                   36.00%  13.5950s       420  32.369ms  27.725ms  74.057ms  new_pos(ParticleSystem*)
                    1.30%  491.53ms     69465  7.0750us  3.5200us  10.112us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                    0.13%  48.002ms         1  48.002ms  48.002ms  48.002ms  init_particle(ParticleSystem*)
                    0.00%  789.48us       421  1.8750us  1.5040us  2.0480us  [CUDA memcpy DtoH]
                    0.00%  2.5600us         2  1.2800us  1.2160us  1.3440us  [CUDA memcpy HtoD]
      API calls:   97.01%  37.2319s      1260  29.549ms  4.7650us  99.870ms  cudaEventSynchronize
                    1.22%  470.13ms     70306  6.6860us  4.0880us  554.32us  cudaLaunchKernel
                    1.04%  399.05ms      1265  315.46us  4.1450us  48.954ms  cudaFree
                    0.37%  141.70ms         1  141.70ms  141.70ms  141.70ms  cudaMemcpyToSymbol
                    0.29%  111.48ms      1265  88.124us  6.4740us  752.27us  cudaMalloc
                    0.03%  12.081ms       422  28.629us  10.817us  700.20us  cudaMemcpy
                    0.02%  6.5330ms      2520  2.5920us  1.2230us  12.258us  cudaEventRecord
                    0.01%  3.8248ms      2520  1.5170us     474ns  472.41us  cudaEventCreate
                    0.01%  2.0886ms      1260  1.6570us     936ns  14.016us  cudaEventElapsedTime
                    0.00%  901.04us        96  9.3850us     276ns  420.56us  cuDeviceGetAttribute
                    0.00%  116.09us         1  116.09us  116.09us  116.09us  cuDeviceTotalMem
                    0.00%  95.840us         1  95.840us  95.840us  95.840us  cuDeviceGetName
                    0.00%  2.4530us         1  2.4530us  2.4530us  2.4530us  cuDeviceGetPCIBusId
                    0.00%  2.3580us         2  1.1790us     481ns  1.8770us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     333ns     767ns  cuDeviceGet
```

##### main_v4.cu
```
==5297== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.37%  413.737s      3664  112.92ms  73.906ms  219.39ms  new_vel_pos(ParticleSystem*)
                    0.61%  2.56046s     10995  232.88us  3.1360us  739.11us  find_min_fitness_parallel(ParticleSystem*, fitness_pos const *, fitness_pos*, int, int, int)
                    0.01%  47.767ms         1  47.767ms  47.767ms  47.767ms  init_particle(ParticleSystem*)
                    0.00%  7.1243ms      3684  1.9330us  1.3440us  2.4000us  [CUDA memcpy DtoH]
                    0.00%  2.5920us         2  1.2960us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   98.82%  413.805s      7328  56.469ms  3.6020us  220.53ms  cudaEventSynchronize
                    0.78%  3.27965s     10995  298.29us  4.5910us  48.405ms  cudaFree
                    0.26%  1.08418s     10997  98.588us  5.7040us  1.0702ms  cudaMalloc
                    0.05%  225.51ms     14661  15.381us  5.1890us  673.26us  cudaLaunchKernel
                    0.03%  143.85ms         1  143.85ms  143.85ms  143.85ms  cudaMemcpyToSymbol
                    0.03%  123.40ms      3685  33.486us  9.9450us  744.87us  cudaMemcpy
                    0.01%  47.221ms     14658  3.2210us  1.1990us  456.61us  cudaEventRecord
                    0.01%  24.653ms     14658  1.6810us     460ns  69.024us  cudaEventCreate
                    0.00%  16.233ms      7328  2.2150us  1.0170us  13.020us  cudaEventElapsedTime
                    0.00%  654.25us        96  6.8150us     250ns  289.93us  cuDeviceGetAttribute
                    0.00%  388.71us         1  388.71us  388.71us  388.71us  cuDeviceGetName
                    0.00%  222.93us         1  222.93us  222.93us  222.93us  cuDeviceTotalMem
                    0.00%  3.2660us         1  3.2660us  3.2660us  3.2660us  cuDeviceGetPCIBusId
                    0.00%  1.6150us         2     807ns     230ns  1.3850us  cuDeviceGetCount
                    0.00%     838ns         2     419ns     198ns     640ns  cuDeviceGet
```

##### main_v5.cu
```
==5785== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   54.99%  24.5096s      1506  16.275ms  14.819ms  74.353ms  calc_fitness(void)
                   43.04%  19.1836s      1506  12.738ms  12.178ms  15.270ms  new_vel_pos(void)
                    1.84%  819.64ms      3014  271.94us  25.440us  559.14us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.11%  50.788ms         1  50.788ms  50.788ms  50.788ms  init_particle(void)
                    0.02%  8.8393ms      4521  1.9550us     832ns  8.1280us  [CUDA memcpy DtoH]
                    0.00%  12.288us        10  1.2280us  1.2160us  1.3440us  [CUDA memcpy HtoD]
      API calls:   95.81%  43.8332s      4518  9.7019ms  1.4210us  74.364ms  cudaEventSynchronize
                    2.41%  1.10272s      3014  365.87us  5.4550us  51.326ms  cudaFree
                    1.19%  543.03ms      3022  179.69us  5.0550us  180.67ms  cudaMalloc
                    0.35%  158.91ms      4521  35.149us  9.1490us  322.34us  cudaMemcpy
                    0.15%  67.876ms      6028  11.260us  5.5250us  393.78us  cudaLaunchKernel
                    0.04%  19.652ms      9038  2.1740us  1.1710us  308.53us  cudaEventRecord
                    0.03%  13.159ms      9038  1.4550us     460ns  34.639us  cudaEventCreate
                    0.01%  6.3529ms      4518  1.4060us     892ns  12.800us  cudaEventElapsedTime
                    0.00%  1.5781ms      6028     261ns     134ns  10.374us  cudaGetSymbolAddress
                    0.00%  851.95us        96  8.8740us     242ns  425.86us  cuDeviceGetAttribute
                    0.00%  410.12us         1  410.12us  410.12us  410.12us  cuDeviceGetName
                    0.00%  111.34us         1  111.34us  111.34us  111.34us  cuDeviceTotalMem
                    0.00%  96.491us        10  9.6490us  8.7540us  14.055us  cudaMemcpyToSymbol
                    0.00%  3.0140us         1  3.0140us  3.0140us  3.0140us  cuDeviceGetPCIBusId
                    0.00%  2.3240us         2  1.1620us     556ns  1.7680us  cuDeviceGetCount
                    0.00%  1.0400us         2     520ns     346ns     694ns  cuDeviceGet

```

##### main_v6.cu
```
==5898== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.15%  22.2500s      1341  15.892ms  15.359ms  21.224ms  calc_fitness(void)
                   42.88%  17.2983s      1341  12.900ms  12.190ms  16.213ms  new_vel_pos(void)
                    1.82%  734.41ms      2684  273.62us  25.440us  560.13us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.13%  50.630ms         1  50.630ms  50.630ms  50.630ms  init_particle(void)
                    0.02%  7.8900ms      4026  1.9590us     832ns  13.792us  [CUDA memcpy DtoH]
                    0.00%  12.288us        10  1.2280us  1.2160us  1.3440us  [CUDA memcpy HtoD]
      API calls:   95.67%  39.6747s      4023  9.8620ms  1.5070us  21.235ms  cudaEventSynchronize
                    2.49%  1.03114s      2684  384.18us  5.1700us  51.185ms  cudaFree
                    1.22%  504.95ms      2692  187.57us  5.0560us  164.44ms  cudaMalloc
                    0.36%  148.51ms      4026  36.887us  8.3160us  947.26us  cudaMemcpy
                    0.16%  68.184ms      5368  12.702us  5.2090us  316.54us  cudaLaunchKernel
                    0.05%  21.387ms      8048  2.6570us  1.1830us  311.84us  cudaEventRecord
                    0.03%  11.523ms      8048  1.4310us     456ns  49.246us  cudaEventCreate
                    0.02%  7.2517ms      4023  1.8020us     921ns  16.555us  cudaEventElapsedTime
                    0.00%  2.0656ms      5368     384ns     142ns  2.1950us  cudaGetSymbolAddress
                    0.00%  895.38us        96  9.3260us     268ns  409.04us  cuDeviceGetAttribute
                    0.00%  399.26us         1  399.26us  399.26us  399.26us  cuDeviceGetName
                    0.00%  109.06us         1  109.06us  109.06us  109.06us  cuDeviceTotalMem
                    0.00%  95.195us        10  9.5190us  8.7910us  13.061us  cudaMemcpyToSymbol
                    0.00%  2.4840us         1  2.4840us  2.4840us  2.4840us  cuDeviceGetPCIBusId
                    0.00%  2.0890us         2  1.0440us     445ns  1.6440us  cuDeviceGetCount
                    0.00%     937ns         2     468ns     295ns     642ns  cuDeviceGet


```
______
### [INPUT 64 DIM, ONE MILION PARTICLES]
##### main_v2.cu
```
==2995== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   58.19%  1.48427s         9  164.92ms  140.22ms  218.32ms  new_vel(ParticleSystem*)
                   36.99%  933.95ms         8  116.74ms  96.016ms  133.09ms  new_pos(ParticleSystem*)
                    3.18%  96.852ms         1  96.852ms  96.852ms  96.852ms  init_particle(ParticleSystem*)
                    1.64%  50.107ms      5985  8.3720us  2.2400us  11.839us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                    0.00%  61.245us        35  1.7490us  1.6640us  1.7920us  [CUDA memcpy DtoH]
                    0.00%  2.5600us         2  1.2800us  1.2160us  1.3440us  [CUDA memcpy HtoD]
      API calls:   91.19%  2.90235s       103  28.178ms  5.2630us  94.179ms  cudaEventSynchronize
                    4.62%  147.16ms       140  1.0512ms  4.3990us  97.839ms  cudaFree
                    2.76%  87.727ms         1  87.727ms  87.727ms  87.727ms  cudaMemcpyToSymbol
                    0.84%  26.813ms      6056  4.4270us     315ns  278.69us  cudaLaunchKernel
                    0.51%  16.087ms       142  113.29us  6.7460us  1.4274ms  cudaMalloc
                    0.03%  1.0456ms        36  29.045us  9.8500us  42.894us  cudaMemcpy
                    0.02%  648.67us        96  6.7560us     182ns  294.55us  cuDeviceGetAttribute
                    0.01%  392.75us       208  1.8880us  1.1150us  7.8110us  cudaEventRecord
                    0.01%  204.54us       208     983ns     497ns  13.508us  cudaEventCreate
                    0.00%  150.36us       103  1.4590us     854ns  10.919us  cudaEventElapsedTime
                    0.00%  66.621us         1  66.621us  66.621us  66.621us  cuDeviceTotalMem
                    0.00%  64.695us         1  64.695us  64.695us  64.695us  cuDeviceGetName
                    0.00%  2.5730us         1  2.5730us  2.5730us  2.5730us  cuDeviceGetPCIBusId
                    0.00%  1.3890us         2     694ns     280ns  1.1090us  cuDeviceGetCount
                    0.00%     729ns         2     364ns     210ns     519ns  cuDeviceGet
```

##### main_v3.cu
```
==2945== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   58.79%  1.48427s         9  164.92ms  140.22ms  218.32ms  new_vel(ParticleSystem*)
                   36.99%  933.95ms         8  116.74ms  96.016ms  133.09ms  new_pos(ParticleSystem*)
                    3.65%  92.241ms         1  92.241ms  92.241ms  92.241ms  init_particle(ParticleSystem*)
                    0.57%  14.421ms      1485  9.7110us  4.6090us  13.443us  find_min_fitness_parallel(ParticleSystem*, fitness_pos*, fitness_pos*, int, int, int)
                    0.00%  14.755us         9  1.6390us  1.6000us  1.7600us  [CUDA memcpy DtoH]
                    0.00%  2.5600us         2  1.2800us  1.2160us  1.3440us  [CUDA memcpy HtoD]
      API calls:   92.01%  2.41843s        25  96.737ms  6.7800us  218.33ms  cudaEventSynchronize
                    4.08%  107.14ms        27  3.9680ms  4.0870us  93.619ms  cudaFree
                    3.46%  90.906ms         1  90.906ms  90.906ms  90.906ms  cudaMemcpyToSymbol
                    0.25%  6.6367ms      1504  4.4120us  3.9670us  21.242us  cudaLaunchKernel
                    0.14%  3.7034ms        29  127.70us  6.5080us  1.4213ms  cudaMalloc
                    0.03%  864.55us        96  9.0050us     252ns  403.48us  cuDeviceGetAttribute
                    0.01%  267.84us        10  26.784us  11.393us  39.300us  cudaMemcpy
                    0.00%  111.43us         1  111.43us  111.43us  111.43us  cuDeviceTotalMem
                    0.00%  109.97us        52  2.1140us  1.4190us  4.8230us  cudaEventRecord
                    0.00%  94.351us         1  94.351us  94.351us  94.351us  cuDeviceGetName
                    0.00%  66.109us        52  1.2710us     465ns  16.149us  cudaEventCreate
                    0.00%  33.796us        25  1.3510us     931ns  2.5540us  cudaEventElapsedTime
                    0.00%  2.8820us         2  1.4410us     972ns  1.9100us  cuDeviceGetCount
                    0.00%  2.3580us         1  2.3580us  2.3580us  2.3580us  cuDeviceGetPCIBusId
                    0.00%  1.1210us         2     560ns     465ns     656ns  cuDeviceGet
```

##### main_v4.cu
```
==2929== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.40%  49.7071s       198  251.05ms  178.75ms  442.45ms  new_vel_pos(ParticleSystem*)
                    0.42%  208.34ms       597  348.98us  3.7750us  1.0710ms  find_min_fitness_parallel(ParticleSystem*, fitness_pos const *, fitness_pos*, int, int, int)
                    0.18%  91.974ms         1  91.974ms  91.974ms  91.974ms  init_particle(ParticleSystem*)
                    0.00%  341.08us       200  1.7050us  1.6310us  1.9510us  [CUDA memcpy DtoH]
                    0.00%  2.5900us         2  1.2950us  1.2150us  1.3750us  [CUDA memcpy HtoD]
      API calls:   98.82%  49.7112s       396  125.53ms  5.7810us  442.48ms  cudaEventSynchronize
                    0.66%  330.28ms       597  553.24us  4.7090us  92.958ms  cudaFree
                    0.39%  196.21ms         1  196.21ms  196.21ms  196.21ms  cudaMemcpyToSymbol
                    0.09%  47.456ms       599  79.225us  5.7340us  1.1885ms  cudaMalloc
                    0.02%  8.6878ms       797  10.900us  5.1950us  39.894us  cudaLaunchKernel
                    0.01%  7.3260ms       201  36.447us  10.031us  45.056us  cudaMemcpy
                    0.00%  1.8492ms       794  2.3290us  1.1840us  17.634us  cudaEventRecord
                    0.00%  990.87us       794  1.2470us     472ns  11.333us  cudaEventCreate
                    0.00%  679.13us        96  7.0740us     219ns  367.56us  cuDeviceGetAttribute
                    0.00%  659.18us       396  1.6640us     989ns  7.9020us  cudaEventElapsedTime
                    0.00%  399.63us         1  399.63us  399.63us  399.63us  cuDeviceGetName
                    0.00%  96.114us         1  96.114us  96.114us  96.114us  cuDeviceTotalMem
                    0.00%  2.2910us         1  2.2910us  2.2910us  2.2910us  cuDeviceGetPCIBusId
                    0.00%  1.9700us         2     985ns     391ns  1.5790us  cuDeviceGetCount
                    0.00%     967ns         2     483ns     257ns     710ns  cuDeviceGet
```

##### main_v5.cu
```
==6214== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.19%  26.7934s       804  33.325ms  28.967ms  117.41ms  calc_fitness(void)
                   42.62%  20.3230s       804  25.277ms  24.306ms  27.152ms  new_vel_pos(void)
                    0.97%  460.99ms      1610  286.33us  45.504us  560.00us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.21%  101.80ms         1  101.80ms  101.80ms  101.80ms  init_particle(void)
                    0.01%  4.8519ms      2415  2.0090us     960ns  4.8320us  [CUDA memcpy DtoH]
                    0.00%  12.480us        10  1.2480us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   97.72%  47.1407s      2412  19.544ms  2.8050us  117.42ms  cudaEventSynchronize
                    1.41%  679.41ms      1610  421.99us  5.7510us  102.39ms  cudaFree
                    0.57%  273.71ms      1618  169.17us  4.8710us  80.146ms  cudaMalloc
                    0.18%  85.152ms      2415  35.259us  10.815us  92.926us  cudaMemcpy
                    0.08%  38.761ms      3220  12.037us  6.3360us  37.729us  cudaLaunchKernel
                    0.02%  11.786ms      4826  2.4420us  1.3230us  334.56us  cudaEventRecord
                    0.01%  5.9239ms      4826  1.2270us     448ns  53.408us  cudaEventCreate
                    0.01%  3.5944ms      2412  1.4900us     953ns  5.6590us  cudaEventElapsedTime
                    0.00%  815.60us      3220     253ns     137ns  1.0100us  cudaGetSymbolAddress
                    0.00%  711.43us        96  7.4100us     214ns  316.18us  cuDeviceGetAttribute
                    0.00%  105.78us         1  105.78us  105.78us  105.78us  cuDeviceTotalMem
                    0.00%  94.531us        10  9.4530us  8.7000us  13.538us  cudaMemcpyToSymbol
                    0.00%  89.056us         1  89.056us  89.056us  89.056us  cuDeviceGetName
                    0.00%  3.5250us         1  3.5250us  3.5250us  3.5250us  cuDeviceGetPCIBusId
                    0.00%  1.7780us         2     889ns     381ns  1.3970us  cuDeviceGetCount
                    0.00%     808ns         2     404ns     235ns     573ns  cuDeviceGet
```

##### main_v6.cu
```
==6232== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   54.53%  6.64677s       210  31.651ms  29.778ms  34.899ms  calc_fitness(void)
                   43.58%  5.31190s       210  25.295ms  24.712ms  27.127ms  new_vel_pos(void)
                    1.04%  126.26ms       422  299.18us  45.569us  558.56us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.85%  103.26ms         1  103.26ms  103.26ms  103.26ms  init_particle(void)
                    0.01%  1.2420ms       633  1.9620us  1.0560us  2.5920us  [CUDA memcpy DtoH]
                    0.00%  12.320us        10  1.2320us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   95.19%  11.9664s       630  18.994ms  3.6570us  34.907ms  cudaEventSynchronize
                    2.43%  305.57ms       430  710.64us  4.8410us  254.14ms  cudaMalloc
                    2.07%  260.24ms       422  616.67us  6.0020us  103.85ms  cudaFree
                    0.18%  22.279ms       633  35.195us  11.302us  84.775us  cudaMemcpy
                    0.08%  9.7308ms       844  11.529us  5.8510us  39.675us  cudaLaunchKernel
                    0.02%  3.0101ms      1262  2.3850us  1.2600us  13.644us  cudaEventRecord
                    0.01%  1.4871ms      1262  1.1780us     473ns  11.131us  cudaEventCreate
                    0.01%  1.0146ms       630  1.6100us     975ns  7.5010us  cudaEventElapsedTime
                    0.00%  507.97us        96  5.2910us     252ns  222.58us  cuDeviceGetAttribute
                    0.00%  253.85us       844     300ns     136ns  5.2540us  cudaGetSymbolAddress
                    0.00%  109.19us         1  109.19us  109.19us  109.19us  cuDeviceTotalMem
                    0.00%  106.82us         1  106.82us  106.82us  106.82us  cuDeviceGetName
                    0.00%  94.742us        10  9.4740us  8.5880us  13.174us  cudaMemcpyToSymbol
                    0.00%  2.5040us         1  2.5040us  2.5040us  2.5040us  cuDeviceGetPCIBusId
                    0.00%  2.2600us         2  1.1300us     415ns  1.8450us  cuDeviceGetCount
                    0.00%     944ns         2     472ns     278ns     666ns  cuDeviceGet

```
