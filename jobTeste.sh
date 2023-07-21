#PBS -N nome
#PBS -l ncpus=1
#PBS -l walltime=24:00:00

#PBS -V
python ~/bioma-tcn-ae/TCNAE_all.py > teste.log
