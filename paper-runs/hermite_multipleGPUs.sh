TREETYPE=1

SFLAG=1
PFLAG=0
DFLAG=0

#N=821000
#N=2365328
N=1000000
#1328096



## Coulomb: Hermite
#OUTFILE=/home/njvaughn/synchronizedDataFiles/KITCpaperData/hermiteTesting/coulomb/K20_hermite_GPU_parallelized_$N.csv
#OUTFILE=/home/njvaughn/synchronizedDataFiles/KITCpaperData/hermiteTesting/coulomb/TitanV_hermite_GPU_parallelized_nonStatic_$N.csv
#OUTFILE=/home/njvaughn/synchronizedDataFiles/KITCpaperData/hermiteTesting/coulomb/bughunt.csv

#OUTFILE=/home/njvaughn/synchronizedDataFiles/KITCpaperData/parallelGPU/TitanV_hermite_parallel_$N.csv 
OUTFILE=/home/njvaughn/synchronizedDataFiles/KITCpaperData/gpu_vs_cpu/TitanV_Coulomb_$N.csv 


SOURCES=/scratch/krasny_fluxg/njvaughn/random/S$N.bin    
TARGETS=/scratch/krasny_fluxg/njvaughn/random/T$N.bin
#SOURCES=/scratch/krasny_fluxg/njvaughn/examplesBenzene/S$N.bin
#TARGETS=/scratch/krasny_fluxg/njvaughn/examplesBenzene/T$N.bin
NUMSOURCES=$N
NUMTARGETS=$N
#DIRECTSUM=/scratch/krasny_fluxg/njvaughn/random/ex_st_coulombSS_$N.bin  
#DIRECTSUM=/scratch/krasny_fluxg/njvaughn/random/ex_st_yukawa_$N.bin  
DIRECTSUM=/scratch/krasny_fluxg/njvaughn/random/ex_st_coulomb_$N.bin  
#DS_CSV=/home/njvaughn/synchronizedDataFiles/KITCpaperData/hermiteTesting/coulomb/TitanV_directSum_GPU_parallelized.csv
DS_CSV=/home/njvaughn/synchronizedDataFiles/KITCpaperData/gpu_vs_cpu/directSum_TitanV_Coulomb.csv 


NUMDEVICES=1
KAPPA=0.0
POTENTIALTYPE=0
../bin/direct.exe   $SOURCES $TARGETS $DIRECTSUM $DS_CSV $N $N $KAPPA $POTENTIALTYPE $NUMDEVICES  


for ORDER in {1..15}
do
	for THETA in 0.4 0.5 0.6 0.7 0.8 0.9
	  do    
		for BATCHSIZE in 5000
		  do       
		     for MAXPARNODE in 5000
		     	do
		     	for POTENTIALTYPE in 0     
		     	do
		     		../bin/tree.exe   $SOURCES $TARGETS $DIRECTSUM $OUTFILE $NUMSOURCES $NUMTARGETS $THETA $ORDER $TREETYPE $MAXPARNODE $KAPPA $POTENTIALTYPE $PFLAG $SFLAG $DFLAG $BATCHSIZE $NUMDEVICES
		     	done
		     done 
		 done
	done
done 

