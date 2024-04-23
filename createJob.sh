#PBS -N TCNAEUCRFalta
#PBS -l select=1:ngpus=1
#PBS -l walltime=300:00:00
#PBS -oe
#PBS -m abe
#PBS -M emerson.okano@unifesp.br
#PBS -V

python ~/bioma-tcn-ae/run_faltaUCR.py >> log/log_falta_TCNAE_int.log
