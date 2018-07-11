
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
==4371== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.29%  13.905ms        10  1.3905ms  1.2740ms  1.8451ms  calc_fitness(void)
                   30.14%  9.0544ms        10  905.44us  899.10us  910.56us  new_vel_pos(void)
                   20.83%  6.2574ms        22  284.43us  7.3910us  561.46us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    2.49%  748.10us         1  748.10us  748.10us  748.10us  init_particle(void)
                    0.22%  64.699us        33  1.9600us  1.3120us  2.2720us  [CUDA memcpy DtoH]
                    0.04%  12.255us        10  1.2250us  1.1840us  1.3440us  [CUDA memcpy HtoD]
      API calls:   76.96%  119.71ms        30  3.9902ms  5.0990us  116.35ms  cudaMalloc
                   15.06%  23.421ms        30  780.70us  6.8230us  1.9990ms  cudaEventSynchronize
                    6.07%  9.4478ms        22  429.45us  5.6120us  1.3548ms  cudaFree
                    0.79%  1.2293ms        33  37.251us  11.611us  86.039us  cudaMemcpy
                    0.48%  739.82us        96  7.7060us     241ns  329.55us  cuDeviceGetAttribute
                    0.30%  465.90us        43  10.834us  5.0920us  24.487us  cudaLaunchKernel
                    0.07%  114.79us        60  1.9130us  1.1130us  13.291us  cudaEventRecord
                    0.07%  103.09us         1  103.09us  103.09us  103.09us  cuDeviceGetName
                    0.07%  102.85us         1  102.85us  102.85us  102.85us  cuDeviceTotalMem
                    0.06%  98.567us        10  9.8560us  9.1360us  13.766us  cudaMemcpyToSymbol
                    0.04%  66.403us        60  1.1060us     493ns  12.131us  cudaEventCreate
                    0.02%  35.432us        30  1.1810us     868ns  3.0200us  cudaEventElapsedTime
                    0.01%  8.6640us        44     196ns     129ns     410ns  cudaGetSymbolAddress
                    0.00%  2.7740us         2  1.3870us     421ns  2.3530us  cuDeviceGetCount
                    0.00%  2.1120us         1  2.1120us  2.1120us  2.1120us  cuDeviceGetPCIBusId
                    0.00%     946ns         2     473ns     287ns     659ns  cuDeviceGet
```

##### main_v6.cu
```
==4397== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.52%  10.097ms        10  1.0097ms  992.42us  1.0204ms  calc_fitness(void)
                   34.50%  9.0421ms        10  904.21us  899.16us  907.29us  new_vel_pos(void)
                   23.84%  6.2484ms        22  284.02us  7.4250us  561.44us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    2.84%  744.91us         1  744.91us  744.91us  744.91us  init_particle(void)
                    0.25%  65.253us        33  1.9770us  1.4400us  2.3050us  [CUDA memcpy DtoH]
                    0.05%  12.321us        10  1.2320us  1.1840us  1.3440us  [CUDA memcpy HtoD]
      API calls:   72.07%  83.342ms        30  2.7781ms  4.5880us  79.839ms  cudaMalloc
                   17.69%  20.459ms        30  681.97us  7.4750us  1.6161ms  cudaEventSynchronize
                    7.73%  8.9436ms        22  406.53us  5.5450us  1.2350ms  cudaFree
                    1.04%  1.2051ms        33  36.518us  11.623us  85.773us  cudaMemcpy
                    0.64%  738.17us        96  7.6890us     236ns  329.21us  cuDeviceGetAttribute
                    0.41%  469.20us        43  10.911us  5.2200us  25.715us  cudaLaunchKernel
                    0.09%  101.03us         1  101.03us  101.03us  101.03us  cuDeviceTotalMem
                    0.09%  100.32us        60  1.6710us  1.0800us  3.3150us  cudaEventRecord
                    0.08%  97.001us        10  9.7000us  9.2730us  12.487us  cudaMemcpyToSymbol
                    0.08%  93.679us         1  93.679us  93.679us  93.679us  cuDeviceGetName
                    0.04%  50.454us        60     840ns     431ns  2.2450us  cudaEventCreate
                    0.03%  32.626us        30  1.0870us     860ns  1.8760us  cudaEventElapsedTime
                    0.01%  8.4910us        44     192ns     130ns     426ns  cudaGetSymbolAddress
                    0.00%  2.5970us         1  2.5970us  2.5970us  2.5970us  cuDeviceGetPCIBusId
                    0.00%  2.0750us         2  1.0370us     335ns  1.7400us  cuDeviceGetCount
                    0.00%  1.0410us         2     520ns     261ns     780ns  cuDeviceGet

```

##### main_v7.cu
```
==4415== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.70%  16.710ms         6  2.7851ms  2.3920ms  3.3282ms  calc_fitness(void)
                   26.68%  8.4611ms         6  1.4102ms  1.3600ms  1.5244ms  new_vel_pos(void)
                   12.56%  3.9834ms        14  284.53us  11.192us  559.12us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    7.87%  2.4954ms         1  2.4954ms  2.4954ms  2.4954ms  init_particle(void)
                    0.15%  46.109us        21  2.1950us  1.3710us  5.6440us  [CUDA memcpy DtoH]
                    0.04%  12.308us        10  1.2300us  1.2110us  1.3710us  [CUDA memcpy HtoD]
      API calls:   77.58%  126.56ms        22  5.7526ms  4.7800us  123.85ms  cudaMalloc
                   16.38%  26.724ms        18  1.4847ms  6.4920us  3.3271ms  cudaEventSynchronize
                    4.61%  7.5232ms        14  537.37us  5.8790us  2.9613ms  cudaFree
                    0.49%  792.56us        21  37.740us  11.483us  93.770us  cudaMemcpy
                    0.43%  706.30us        96  7.3570us     235ns  315.87us  cuDeviceGetAttribute
                    0.20%  329.84us        27  12.216us  5.6830us  27.919us  cudaLaunchKernel
                    0.07%  115.86us        36  3.2180us  1.1240us  12.241us  cudaEventRecord
                    0.06%  98.102us         1  98.102us  98.102us  98.102us  cuDeviceGetName
                    0.06%  97.980us        10  9.7980us  9.1030us  13.037us  cudaMemcpyToSymbol
                    0.06%  91.584us         1  91.584us  91.584us  91.584us  cuDeviceTotalMem
                    0.02%  38.451us        36  1.0680us     474ns  3.0820us  cudaEventCreate
                    0.01%  24.153us        18  1.3410us     919ns  2.6810us  cudaEventElapsedTime
                    0.01%  12.659us         1  12.659us  12.659us  12.659us  cuDeviceGetPCIBusId
                    0.00%  5.9170us        28     211ns     138ns     441ns  cudaGetSymbolAddress
                    0.00%  2.1790us         2  1.0890us     315ns  1.8640us  cuDeviceGetCount
                    0.00%     860ns         2     430ns     272ns     588ns  cuDeviceGet

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
==4063== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.63%  4.11940s       841  4.8982ms  3.9070ms  14.249ms  calc_fitness(void)
                   38.17%  2.82617s       842  3.3565ms  3.1526ms  4.6683ms  new_vel_pos(void)
                    6.02%  446.01ms      1684  264.85us  10.446us  561.21us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.11%  8.2482ms         1  8.2482ms  8.2482ms  8.2482ms  init_particle(void)
                    0.07%  5.0323ms      2526  1.9920us     961ns  7.4340us  [CUDA memcpy DtoH]
                    0.00%  12.336us        10  1.2330us  1.2170us  1.3770us  [CUDA memcpy HtoD]
      API calls:   87.37%  6.99890s      2524  2.7729ms  1.9770us  14.237ms  cudaEventSynchronize
                    7.41%  593.60ms      1684  352.49us  5.2360us  8.7367ms  cudaFree
                    3.45%  276.12ms      1692  163.19us  4.5730us  75.587ms  cudaMalloc
                    1.11%  89.305ms      2526  35.354us  8.6520us  95.638us  cudaMemcpy
                    0.42%  33.380ms      3369  9.9080us  4.7880us  35.730us  cudaLaunchKernel
                    0.12%  9.9775ms      5050  1.9750us  1.1260us  297.32us  cudaEventRecord
                    0.06%  4.9302ms      5050     976ns     432ns  40.585us  cudaEventCreate
                    0.04%  3.0745ms      2524  1.2180us     849ns  4.1540us  cudaEventElapsedTime
                    0.01%  739.40us      3368     219ns     130ns     606ns  cudaGetSymbolAddress
                    0.01%  578.75us        96  6.0280us     142ns  258.77us  cuDeviceGetAttribute
                    0.00%  95.415us        10  9.5410us  8.9610us  12.651us  cudaMemcpyToSymbol
                    0.00%  84.883us         1  84.883us  84.883us  84.883us  cuDeviceGetName
                    0.00%  71.815us         1  71.815us  71.815us  71.815us  cuDeviceTotalMem
                    0.00%  2.3640us         1  2.3640us  2.3640us  2.3640us  cuDeviceGetPCIBusId
                    0.00%  1.3730us         2     686ns     297ns  1.0760us  cuDeviceGetCount
                    0.00%     615ns         2     307ns     161ns     454ns  cuDeviceGet
```

##### main_v6.cu
```
==4093== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.36%  327.28ms        62  5.2786ms  4.8011ms  6.4107ms  calc_fitness(void)
                   36.27%  210.61ms        62  3.3970ms  3.1612ms  4.7496ms  new_vel_pos(void)
                    5.89%  34.215ms       126  271.55us  10.445us  564.05us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    1.42%  8.2227ms         1  8.2227ms  8.2227ms  8.2227ms  init_particle(void)
                    0.07%  384.30us       189  2.0330us  1.3140us  8.3950us  [CUDA memcpy DtoH]
                    0.00%  12.306us        10  1.2300us  1.1210us  1.3780us  [CUDA memcpy HtoD]
      API calls:   72.51%  541.32ms       186  2.9103ms  6.2270us  6.4201ms  cudaEventSynchronize
                   18.91%  141.19ms       134  1.0536ms  4.7230us  125.20ms  cudaMalloc
                    6.98%  52.126ms       126  413.70us  5.3330us  8.7306ms  cudaFree
                    0.91%  6.7698ms       189  35.819us  11.337us  86.285us  cudaMemcpy
                    0.37%  2.7643ms       251  11.013us  4.8880us  24.430us  cudaLaunchKernel
                    0.11%  791.07us        96  8.2400us     311ns  344.66us  cuDeviceGetAttribute
                    0.09%  706.81us       372  1.9000us  1.1540us  13.263us  cudaEventRecord
                    0.05%  336.33us       372     904ns     430ns  12.172us  cudaEventCreate
                    0.03%  199.48us       186  1.0720us     836ns  2.3470us  cudaEventElapsedTime
                    0.01%  107.55us         1  107.55us  107.55us  107.55us  cuDeviceTotalMem
                    0.01%  98.825us        10  9.8820us  9.3840us  13.310us  cudaMemcpyToSymbol
                    0.01%  98.799us         1  98.799us  98.799us  98.799us  cuDeviceGetName
                    0.01%  48.428us       252     192ns     138ns     526ns  cudaGetSymbolAddress
                    0.00%  2.8330us         2  1.4160us     577ns  2.2560us  cuDeviceGetCount
                    0.00%  2.4610us         1  2.4610us  2.4610us  2.4610us  cuDeviceGetPCI
```

##### main_v7.cu
```
==4113== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   59.13%  3.94763s      1486  2.6565ms  2.5020ms  4.1502ms  new_vel_pos(void)
                   28.87%  1.92760s      1486  1.2972ms  1.0661ms  5.0118ms  calc_fitness(void)
                   11.75%  784.73ms      2974  263.86us  10.405us  561.42us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.13%  8.8063ms      4461  1.9740us     832ns  7.4920us  [CUDA memcpy DtoH]
                    0.12%  7.9399ms         1  7.9399ms  7.9399ms  7.9399ms  init_particle(void)
                    0.00%  12.294us        10  1.2290us  1.2160us  1.3450us  [CUDA memcpy HtoD]
      API calls:   76.82%  6.00064s      4458  1.3460ms  2.2400us  5.0218ms  cudaEventSynchronize
                   13.32%  1.04072s      2974  349.94us  5.4040us  8.4422ms  cudaFree
                    6.47%  505.37ms      2982  169.47us  5.0750us  132.20ms  cudaMalloc
                    2.18%  170.47ms      4461  38.213us  9.0030us  815.04us  cudaMemcpy
                    0.75%  58.655ms      5948  9.8610us  5.0130us  388.10us  cudaLaunchKernel
                    0.23%  17.751ms      8918  1.9900us  1.0680us  306.63us  cudaEventRecord
                    0.12%  9.3465ms      8918  1.0480us     439ns  36.722us  cudaEventCreate
                    0.07%  5.6708ms      4458  1.2720us     847ns  11.123us  cudaEventElapsedTime
                    0.02%  1.4037ms      5948     236ns     131ns     686ns  cudaGetSymbolAddress
                    0.01%  742.85us        96  7.7370us     274ns  328.53us  cuDeviceGetAttribute
                    0.00%  103.96us         1  103.96us  103.96us  103.96us  cuDeviceGetName
                    0.00%  102.30us         1  102.30us  102.30us  102.30us  cuDeviceTotalMem
                    0.00%  97.046us        10  9.7040us  9.0450us  13.118us  cudaMemcpyToSymbol
                    0.00%  2.9050us         1  2.9050us  2.9050us  2.9050us  cuDeviceGetPCIBusId
                    0.00%  1.9170us         2     958ns     432ns  1.4850us  cuDeviceGetCount
                    0.00%     977ns         2     488ns     355ns     622ns  cuDeviceGet
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
==3742== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   54.07%  3.67447s       437  8.4084ms  6.8366ms  34.772ms  calc_fitness(void)
                   42.17%  2.86573s       438  6.5428ms  6.1527ms  8.5098ms  new_vel_pos(void)
                    3.44%  233.78ms       876  266.87us  15.410us  560.99us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.29%  19.440ms         1  19.440ms  19.440ms  19.440ms  init_particle(void)
                    0.04%  2.6331ms      1314  2.0030us     959ns  7.2900us  [CUDA memcpy DtoH]
                    0.00%  12.342us        10  1.2340us  1.2150us  1.3750us  [CUDA memcpy HtoD]
      API calls:   91.03%  6.58235s      1312  5.0170ms  2.1280us  34.770ms  cudaEventSynchronize
                    4.57%  330.68ms       876  377.49us  5.4070us  19.956ms  cudaFree
                    3.31%  239.47ms       884  270.89us  4.6900us  148.27ms  cudaMalloc
                    0.64%  46.340ms      1314  35.266us  9.3000us  92.643us  cudaMemcpy
                    0.26%  19.135ms      1753  10.915us  5.6550us  37.548us  cudaLaunchKernel
                    0.09%  6.6308ms      2626  2.5250us  1.1830us  304.64us  cudaEventRecord
                    0.04%  2.8770ms      2626  1.0950us     491ns  26.734us  cudaEventCreate
                    0.02%  1.5938ms      1312  1.2140us     880ns  3.1530us  cudaEventElapsedTime
                    0.01%  842.49us        96  8.7750us     227ns  390.12us  cuDeviceGetAttribute
                    0.01%  392.07us         1  392.07us  392.07us  392.07us  cuDeviceGetName
                    0.00%  352.96us      1752     201ns     130ns  9.0990us  cudaGetSymbolAddress
                    0.00%  104.92us         1  104.92us  104.92us  104.92us  cuDeviceTotalMem
                    0.00%  95.265us        10  9.5260us  8.8310us  13.785us  cudaMemcpyToSymbol
                    0.00%  2.2900us         1  2.2900us  2.2900us  2.2900us  cuDeviceGetPCIBusId
                    0.00%  2.0710us         2  1.0350us     406ns  1.6650us  cuDeviceGetCount
                    0.00%  1.1200us         2     560ns     255ns     865ns  cuDeviceGet
```

##### main_v6.cu
```
==3464== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   53.22%  1.20952s       148  8.1724ms  7.2739ms  11.694ms  calc_fitness(void)
                   42.33%  962.16ms       148  6.5011ms  6.1541ms  8.0333ms  new_vel_pos(void)
                    3.55%  80.686ms       298  270.76us  15.454us  561.96us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.86%  19.495ms         1  19.495ms  19.495ms  19.495ms  init_particle(void)
                    0.04%  889.67us       447  1.9900us     960ns  7.5830us  [CUDA memcpy DtoH]
                    0.00%  12.256us        10  1.2250us  1.1840us  1.3440us  [CUDA memcpy HtoD]
      API calls:   88.72%  2.18814s       444  4.9282ms  3.4950us  11.703ms  cudaEventSynchronize
                    5.14%  126.75ms       298  425.32us  5.1460us  20.019ms  cudaFree
                    5.07%  125.00ms       306  408.50us  4.6240us  91.561ms  cudaMalloc
                    0.67%  16.565ms       447  37.057us  9.1750us  847.52us  cudaMemcpy
                    0.22%  5.3988ms       595  9.0730us  5.4120us  31.323us  cudaLaunchKernel
                    0.07%  1.7553ms       888  1.9760us  1.1520us  16.086us  cudaEventRecord
                    0.03%  800.41us       888     901ns     433ns  11.927us  cudaEventCreate
                    0.03%  775.59us        96  8.0790us     215ns  362.38us  cuDeviceGetAttribute
                    0.02%  486.10us       444  1.0940us     851ns  10.439us  cudaEventElapsedTime
                    0.02%  391.44us         1  391.44us  391.44us  391.44us  cuDeviceGetName
                    0.00%  115.96us       596     194ns     136ns     535ns  cudaGetSymbolAddress
                    0.00%  98.303us         1  98.303us  98.303us  98.303us  cuDeviceTotalMem
                    0.00%  94.739us        10  9.4730us  8.9380us  12.953us  cudaMemcpyToSymbol
                    0.00%  2.2810us         1  2.2810us  2.2810us  2.2810us  cuDeviceGetPCIBusId
                    0.00%  1.9350us         2     967ns     297ns  1.6380us  cuDeviceGetCount
                    0.00%     844ns         2     422ns     238ns     606ns  cuDeviceGet
```

##### main_v7.cu
```
==3432== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   69.86%  79.9386s     15478  5.1647ms  4.7688ms  8.1129ms  new_vel_pos(void)
                   22.84%  26.1400s     15477  1.6890ms  1.6199ms  8.3765ms  calc_fitness(void)
                    7.20%  8.23896s     30956  266.15us  15.454us  560.86us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.08%  90.441ms     46434  1.9470us     831ns  9.3430us  [CUDA memcpy DtoH]
                    0.02%  18.983ms         1  18.983ms  18.983ms  18.983ms  init_particle(void)
                    0.00%  12.287us        10  1.2280us  1.2150us  1.3440us  [CUDA memcpy HtoD]
      API calls:   86.38%  107.226s     46432  2.3093ms  1.3430us  8.3851ms  cudaEventSynchronize
                    8.48%  10.5233s     30956  339.94us  5.2460us  19.507ms  cudaFree
                    2.95%  3.66240s     30964  118.28us  4.7140us  143.26ms  cudaMalloc
                    1.33%  1.64749s     46434  35.480us  8.3510us  782.60us  cudaMemcpy
                    0.54%  667.87ms     61913  10.787us  5.6530us  710.13us  cudaLaunchKernel
                    0.17%  209.45ms     92866  2.2550us  1.1180us  386.92us  cudaEventRecord
                    0.09%  109.06ms     92866  1.1740us     473ns  673.76us  cudaEventCreate
                    0.05%  65.203ms     46432  1.4040us     868ns  380.06us  cudaEventElapsedTime
                    0.01%  14.896ms     61912     240ns     123ns  9.9510us  cudaGetSymbolAddress
                    0.00%  800.58us        96  8.3390us     338ns  351.20us  cuDeviceGetAttribute
                    0.00%  119.16us         1  119.16us  119.16us  119.16us  cuDeviceTotalMem
                    0.00%  98.197us         1  98.197us  98.197us  98.197us  cuDeviceGetName
                    0.00%  95.029us        10  9.5020us  8.5010us  13.399us  cudaMemcpyToSymbol
                    0.00%  3.2090us         1  3.2090us  3.2090us  3.2090us  cuDeviceGetPCIBusId
                    0.00%  2.3920us         2  1.1960us     478ns  1.9140us  cuDeviceGetCount
                    0.00%  1.2180us         2     609ns     429ns     789ns  cuDeviceGet
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

##### main_v7.cu
```
==3121== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   69.77%  12.5944s      1269  9.9247ms  9.2951ms  12.749ms  new_vel_pos(void)
                   26.08%  4.70733s      1268  3.7124ms  3.3088ms  11.502ms  calc_fitness(void)
                    3.82%  690.25ms      2538  271.97us  25.487us  557.24us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.29%  51.700ms         1  51.700ms  51.700ms  51.700ms  init_particle(void)
                    0.04%  7.5367ms      3807  1.9790us     831ns  8.3140us  [CUDA memcpy DtoH]
                    0.00%  12.282us        10  1.2280us  1.2150us  1.3430us  [CUDA memcpy HtoD]
      API calls:   91.36%  17.4511s      3805  4.5864ms  2.2230us  12.762ms  cudaEventSynchronize
                    5.06%  967.07ms      2538  381.04us  5.3330us  52.250ms  cudaFree
                    2.39%  455.85ms      2546  179.05us  4.9980us  165.88ms  cudaMalloc
                    0.72%  137.19ms      3807  36.036us  9.4850us  771.87us  cudaMemcpy
                    0.27%  51.127ms      5077  10.070us  5.1580us  109.09us  cudaLaunchKernel
                    0.09%  16.709ms      7612  2.1950us  1.0720us  319.46us  cudaEventRecord
                    0.08%  14.661ms      7612  1.9260us     451ns  45.803us  cudaEventCreate
                    0.03%  4.7872ms      3805  1.2580us     883ns  12.651us  cudaEventElapsedTime
                    0.01%  1.0182ms      5076     200ns     128ns  10.346us  cudaGetSymbolAddress
                    0.00%  531.01us        96  5.5310us     134ns  280.33us  cuDeviceGetAttribute
                    0.00%  180.13us         1  180.13us  180.13us  180.13us  cuDeviceTotalMem
                    0.00%  107.37us         1  107.37us  107.37us  107.37us  cuDeviceGetName
                    0.00%  95.691us        10  9.5690us  8.9760us  13.639us  cudaMemcpyToSymbol
                    0.00%  2.8420us         1  2.8420us  2.8420us  2.8420us  cuDeviceGetPCIBusId
                    0.00%  1.5740us         2     787ns     198ns  1.3760us  cuDeviceGetCount
                    0.00%     634ns         2     317ns     171ns     463ns  cuDeviceGet
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
##### main_v7.cu
```
==2740== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.60%  12.6369s       649  19.471ms  18.393ms  20.671ms  new_vel_pos(void)
                   28.80%  5.30597s       648  8.1882ms  7.4247ms  15.337ms  calc_fitness(void)
                    2.01%  370.73ms      1298  285.62us  45.568us  557.44us  find_min_fitness_parallel(fitness_pos const *, fitness_pos*, int, int, int)
                    0.56%  102.85ms         1  102.85ms  102.85ms  102.85ms  init_particle(void)
                    0.02%  4.0293ms      1947  2.0690us     832ns  5.9520us  [CUDA memcpy DtoH]
                    0.00%  12.352us        10  1.2350us  1.2160us  1.3760us  [CUDA memcpy HtoD]
      API calls:   94.03%  17.9855s      1945  9.2471ms  2.1280us  20.681ms  cudaEventSynchronize
                    3.03%  579.42ms      1298  446.40us  5.2230us  103.45ms  cudaFree
                    2.38%  455.01ms      1306  348.40us  4.7640us  318.32ms  cudaMalloc
                    0.36%  68.623ms      1947  35.245us  9.1370us  86.632us  cudaMemcpy
                    0.12%  23.324ms      2597  8.9810us  5.3260us  39.359us  cudaLaunchKernel
                    0.04%  7.7489ms      3892  1.9900us  1.1470us  299.09us  cudaEventRecord
                    0.02%  3.6402ms      3892     935ns     430ns  24.843us  cudaEventCreate
                    0.01%  2.3062ms      1945  1.1850us     869ns  11.388us  cudaEventElapsedTime
                    0.00%  881.99us        96  9.1870us     252ns  421.86us  cuDeviceGetAttribute
                    0.00%  695.69us      2596     267ns     131ns     926ns  cudaGetSymbolAddress
                    0.00%  398.53us         1  398.53us  398.53us  398.53us  cuDeviceGetName
                    0.00%  113.16us         1  113.16us  113.16us  113.16us  cuDeviceTotalMem
                    0.00%  94.618us        10  9.4610us  8.8790us  13.569us  cudaMemcpyToSymbol
                    0.00%  2.4420us         1  2.4420us  2.4420us  2.4420us  cuDeviceGetPCIBusId
                    0.00%  2.3340us         2  1.1670us     454ns  1.8800us  cuDeviceGetCount
                    0.00%     978ns         2     489ns     285ns     693ns  cuDeviceGet

```
