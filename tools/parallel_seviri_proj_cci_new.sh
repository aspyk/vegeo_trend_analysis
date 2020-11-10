#!/bin/bash
if [ "$HOSTNAME" == "sxvgo1" ]
   then
   echo "Error your running machine should be aneto, use sqsub to run this script"
   exit
fi
 
DIR=/cnrm/vegeo/SAT/CODES_suman/From_Suman/clean_trends
# NOTE : this would not work : export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#        I believe that this is because aneto copy the script somewhere else. Paths should be absolute.

LOGFILE=$DIR/log/run.aneto_test.log
mkdir -p $DIR/log


###################################################################################################
## START BOILERPLATE.  "Boilerplate code" is any seemingly repetitive code that shows up again and again in order to get some result that seems like it ought to be much simpler.
echo "started" > $LOGFILE

cd $DIR
waitforjobs() {
    # from https://stackoverflow.com/questions/1537956/bash-limit-the-number-of-concurrent-jobs
    # usage : waitforjobs 5 : waits until there are less than 5 backgroung jobs running
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}
date                                                            >> $LOGFILE
echo "Setting conda default environment"                        1>>$LOGFILE 2>>$LOGFILE
. /home/pardem/miniconda3/etc/profile.d/conda.sh                      1>>$LOGFILE 2>>$LOGFILE
conda info --envs                                               1>>$LOGFILE 2>>$LOGFILE;

HDF5_USE_FILE_LOCKING=FALSE;
echo "HDF5_USE_FILE_LOCKING = $HDF5_USE_FILE_LOCKING"           1>>$LOGFILE 2>>$LOGFILE;

date                                                            >> $LOGFILE
echo "Running conda activate"
conda activate  shared_2019_09_21_lustre                                     1>>$LOGFILE 2>>$LOGFILE;

echo '--------------------------'  1>>$LOGFILE 2>>$LOGFILE;  
PATH=/home/pardem/python-team/bin:$PATH                                1>>$LOGFILE 2>>$LOGFILE;
echo "PATH=$PATH"                                               1>>$LOGFILE 2>>$LOGFILE
#PYTHONPATH=$HOME/python-team/custom-python-libraries:$PYTHONPATH 1>>$LOGFILE 2>>$LOGFILE;
#ls $HOME/python-team/custom-python-libraries/*.py   1>>$LOGFILE 2>>$LOGFILE; 
#echo "PYTHONPATH=$PYTHONPATH"                                   1>>$LOGFILE 2>>$LOGFILE
echo '--------------------------'  1>>$LOGFILE 2>>$LOGFILE;
date                                                            1>>$LOGFILE 2>>$LOGFILE
echo "Python is using this conda env : (on "$HOSTNAME" "      1>>$LOGFILE 2>>$LOGFILE
which python                                                    1>>$LOGFILE 2>>$LOGFILE
python -c 'import sys; print(sys.path)'                         1>>$LOGFILE 2>>$LOGFILE
## END BOILERPLATE
###################################################################################################

export LIMIT=20 #$(cat /proc/cpuinfo | grep processor | wc -l)

echo "Anteto initial experiment" > $LOGFILE
python3 $DIR/compute_seviri_projection_cci.py --output './output_cci_chunks' --xlim1 '0' --xlim2 '800' --ylim1 '1500' --ylim2 '3500' -l info 1>>$LOGFILE 2>>$LOGFILE

