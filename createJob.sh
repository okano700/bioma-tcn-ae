#PBS -N TCNAE_UCR
#PBS -l select=1:ngpu=1
#PBS -l walltime=300:00:00
#PBS -oe
#PBS -m abe
#PBS -M emerson.okano@unifesp.br
#PBS -V

python ~/bioma-tcn-ae/run_UCR.py --path ~/datsets/UCR_Anomaly_FullData/\*.txt --n 249
